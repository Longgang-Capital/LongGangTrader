// src/utils.rs

use std::collections::HashMap;
use polars::prelude::*;
use std::path::Path;
use serde::Deserialize;
use std::io::Read;
use pyo3::prelude::*;
use std::fs::File;
use serde_json;
use serde_pickle;

// 导入父模块的 BacktestConfig
use super::BacktestConfig;

/// 辅助函数，将 DataFrame 中各种类型（Datetime, Date, String）的日期列统一转换为 "%Y-%m-%d" 格式的字符串列
fn get_date_column_as_string(df: &DataFrame, col_name: &str) -> PolarsResult<StringChunked> {
    let date_series = df.column(col_name)?;
    match date_series.dtype() {
        DataType::Datetime(_, _) => date_series.datetime()?.to_string("%Y-%m-%d"),
        DataType::Date => date_series.date()?.to_string("%Y-%m-%d"),
        DataType::String => {
            // 尝试将字符串类型转换为日期类型，再格式化为字符串
            // Polars 的 cast 很强大，能自动解析多种日期格式
            let date_series = date_series.cast(&DataType::Date)?;
            date_series.date()?.to_string("%Y-%m-%d")
        },
        other => Err(PolarsError::InvalidOperation(
            format!(
                "日期列 '{}' 的数据类型不受支持: {:?}. 可接受的类型是: Datetime, Date, or String.",
                col_name, other
            )
            .into(),
        )),
    }
}

// --- 公共数据结构 ---

/// pub 关键字使其对 lib.rs 可见
#[derive(Debug)] // 添加 Debug trait 方便调试
pub struct Trade {
    pub symbol: String,
    pub shares: f64,
    pub price: f64,
}

/// pub 关键字使其对 lib.rs 可见
pub struct PortfolioSnapshot {
    pub date: String,
    pub equity: f64,
    pub cash: f64,
    pub holdings_value: f64,
    pub turnover_rate: f64,
}

// --- 公共类型别名 ---

/// pub 关键字使其对 lib.rs 可见
pub type DailyMarketData = HashMap<String, f64>;
pub type DailySignalData = HashMap<String, f64>;
pub type PositionMap = HashMap<String, f64>;


// --- 新增的数据预处理函数 ---

/// 将输入的 market_data DataFrame 转换为 HashMap, 以便按日期快速访问
pub fn preprocess_market_data(df: &DataFrame, config: &BacktestConfig) -> PolarsResult<HashMap<String, DailyMarketData>> {
    let mut map = HashMap::new();
    let date_col = get_date_column_as_string(df, &config.date_col)?;
    let symbol_col = df.column(&config.symbol_col)?.str()?;
    let close_col = df.column(&config.close_col)?.f64()?;

    for i in 0..df.height() {
        let date = date_col.get(i).unwrap().to_string();
        let symbol = symbol_col.get(i).unwrap().to_string();
        let close = close_col.get(i).unwrap();
        
        map.entry(date)
           .or_insert_with(HashMap::new)
           .insert(symbol, close);
    }
    Ok(map)
}

/// 将输入的 signals DataFrame 转换为 HashMap, 以便按日期快速访问
pub fn preprocess_signals(df: &DataFrame, config: &BacktestConfig) -> PolarsResult<HashMap<String, DailySignalData>> {
    let mut map = HashMap::new();
    let date_col = get_date_column_as_string(df, &config.date_col)?;
    let symbol_col = df.column(&config.symbol_col)?.str()?;
    let weight_col = df.column(&config.weight_col)?.f64()?;

    for i in 0..df.height() {
        let date = date_col.get(i).unwrap().to_string();
        let symbol = symbol_col.get(i).unwrap().to_string();
        let weight = weight_col.get(i).unwrap();
        
        map.entry(date)
           .or_insert_with(HashMap::new)
           .insert(symbol, weight);
    }
    Ok(map)
}

/// 从市场数据中获取所有排序且唯一的日期
pub fn get_sorted_unique_dates(df: &DataFrame, config: &BacktestConfig) -> PolarsResult<Vec<String>> {
    let date_col = get_date_column_as_string(df, &config.date_col)?;
    let mut unique_dates: Vec<String> = date_col.unique()?.into_iter().filter_map(
        |opt| opt.map(|s| s.to_string())
    ).collect();
    unique_dates.sort();
    Ok(unique_dates)
}


// --- 公共辅助函数 ---

/// 计算当前持仓的总市值 (Mark-to-Market)
pub fn calculate_holdings_value(
    current_positions: &PositionMap,
    market_data_for_date: &DailyMarketData,
) -> f64 {
    // 示例实现
    current_positions.iter()
        .map(|(symbol, &shares)| {
            let price = market_data_for_date.get(symbol).unwrap_or(&0.0);
            shares * price
        })
        .sum()
}

/// 从预处理好的信号数据中，获取指定日期的目标持仓权重
pub fn get_target_weights_for_date<'a>(
    signals_data: &'a HashMap<String, DailySignalData>,
    date: &str,
) -> Option<&'a DailySignalData> {
    signals_data.get(date)
}

/// 根据总权益和目标权重，计算每只股票的目标持仓市值
pub fn calculate_target_positions_value(
    total_equity: f64,
    target_weights: &DailySignalData,
) -> HashMap<String, f64> {
    target_weights.iter()
        .map(
            |(symbol, &weight)| (symbol.clone(), total_equity * weight)
        )
        .collect()
}

/// 比较当前持仓和目标持仓，生成具体的交易指令列表
pub fn calculate_trades(
    current_positions: &PositionMap,
    target_positions_value: &HashMap<String, f64>,
    market_data_for_date: &DailyMarketData,
) -> Vec<Trade> {
    let mut trades = Vec::new();
    let mut all_symbols: Vec<_> = current_positions.keys().collect();
    all_symbols.extend(target_positions_value.keys());
    all_symbols.sort();
    all_symbols.dedup();

    for symbol in all_symbols {
        let current_shares = *current_positions.get(symbol).unwrap_or(&0.0);
        let target_value = *target_positions_value.get(symbol).unwrap_or(&0.0);
        let price = *market_data_for_date.get(symbol).unwrap_or(&0.0);

        if price > 0.0 {
            let target_shares = target_value / price;
            let shares_to_trade = target_shares - current_shares;

            // 只有在交易股数变化显著时才生成交易
            if shares_to_trade.abs() > 1e-6 {
                trades.push(Trade {
                    symbol: symbol.clone(),
                    shares: shares_to_trade,
                    price,
                });
            }
        }
    }
    trades
}


/// 将记录的投资组合历史快照列表转换为 Polars DataFrame
pub fn build_results_dataframe_from_history(
    portfolio_history: &[PortfolioSnapshot],
) -> PolarsResult<DataFrame> {
    // 从快照中提取列数据
    let dates: Vec<&str> = portfolio_history.iter().map(
        |s| s.date.as_str()
    ).collect();
    let equities: Vec<f64> = portfolio_history.iter().map(
        |s| s.equity
    ).collect();
    let cash: Vec<f64> = portfolio_history.iter().map(
        |s| s.cash
    ).collect();
    let holdings_values: Vec<f64> = portfolio_history.iter().map(
        |s| s.holdings_value
    ).collect();
    let turnover_rates: Vec<f64> = portfolio_history.iter().map(
        |s| s.turnover_rate
    ).collect();

    // 创建 Polars Series
    let date_series = Series::new("date".into(), dates);
    let equity_series = Series::new("equity".into(), equities);
    let cash_series = Series::new("cash".into(), cash);
    let holdings_value_series = Series::new("holdings_value".into(), holdings_values);
    let turnover_rate_series = Series::new("turnover_rate".into(), turnover_rates);

    // 组合成 DataFrame
    DataFrame::new(vec![date_series.into(), equity_series.into(), cash_series.into(), holdings_value_series.into(), turnover_rate_series.into()])
}

/// 根据当天的交易列表和总资产计算换手率
pub fn calculate_turnover_rate(
    trades: &Vec<Trade>,
    total_equity: f64,
) -> f64 {
    if total_equity == 0.0 {
        return 0.0;
    }

    let total_traded_value: f64 = trades.iter()
        .map(|trade| (trade.shares * trade.price).abs())
        .sum();
    
    total_traded_value / total_equity
}

#[derive(Deserialize, Debug)]
pub struct BinFileMetadata {
    dtype: String,
    shape: Vec<usize>,
}

pub fn load_market_data(market_data_path: &str) -> PyResult<DataFrame> {
    match Path::new(market_data_path).extension().and_then(|s| s.to_str()) {
        Some("parquet") => {
            let file = File::open(market_data_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            ParquetReader::new(file).finish().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        },
        Some("csv") => {
            let file = File::open(market_data_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            CsvReader::new(file).finish().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        },
        Some("bin") => {
            let json_path = Path::new(market_data_path).with_extension("bin.json");
            let json_file = File::open(&json_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open companion JSON file: {}", e)))?;
            let metadata: BinFileMetadata = serde_json::from_reader(json_file).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse JSON metadata: {}", e)))?;

            if metadata.dtype != "<f4" {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported .bin dtype, only '<f4' (f32) supported.".to_string()));
            }
            if metadata.shape.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unsupported .bin shape, only 2D arrays supported.".to_string()));
            }

            let mut file = File::open(market_data_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            let mut buffer: Vec<u8> = Vec::new();
            file.read_to_end(&mut buffer).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            
            let data: Vec<f32> = buffer.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect();
            
            let cols = metadata.shape[1];
            let series: Vec<Series> = (0..cols).map(|i| {
                let col_data: Vec<f32> = data.iter().skip(i).step_by(cols).cloned().collect();
                Series::new(format!("col_{}", i).into(), col_data)
            }).collect();

            DataFrame::new(series.into_iter().map(|s| s.into()).collect())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        },
        Some("pkl") => {
            let file = File::open(market_data_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            let data: Vec<HashMap<String, serde_pickle::Value>> = serde_pickle::from_reader(file).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to deserialize .pkl file: {}", e)))?;
            
            if data.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Pickle file is empty or in an unsupported format.".to_string()));
            }

            let columns: Vec<String> = data[0].keys().cloned().collect();
            let mut series_vec: Vec<Series> = Vec::new();

            for col in &columns {
                let mut values: Vec<Option<f64>> = Vec::new();
                for record in &data {
                    if let Some(val) = record.get(col) {
                        match val {
                            serde_pickle::Value::F64(f) => values.push(Some(*f)),
                            serde_pickle::Value::I64(i) => values.push(Some(*i as f64)),
                            _ => values.push(None)
                        }
                    } else {
                        values.push(None);
                    }
                }
                series_vec.push(Series::new(col.into(), values));
            }
            DataFrame::new(series_vec.into_iter().map(|s| s.into()).collect())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        },
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unsupported file extension: {:?}", market_data_path))),
    }
}