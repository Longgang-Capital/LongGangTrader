// src/utils.rs

use std::collections::HashMap;
use polars::prelude::*;

// 导入父模块的 BacktestConfig
use super::BacktestConfig;

/// 辅助函数，将 DataFrame 中各种类型（Datetime, Date, String）的日期列统一转换为 "%Y-%m-%d" 格式的字符串列
fn get_date_column_as_string(df: &DataFrame, col_name: &str) -> PolarsResult<StringChunked> {
    let date_series = df.column(col_name)?;
    match date_series.dtype() {
        DataType::Datetime(_, _) => date_series.datetime()?.to_string("%Y-%m-%d"),
        DataType::Date => Ok(date_series.date()?.to_string("%Y-%m-%d")),
        DataType::String => {
            // 尝试将字符串类型转换为日期类型，再格式化为字符串
            // Polars 的 cast 很强大，能自动解析多种日期格式
            let date_series = date_series.cast(&DataType::Date)?;
            Ok(date_series.date()?.to_string("%Y-%m-%d"))
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

    // 创建 Polars Series
    let date_series = Series::new("date", dates);
    let equity_series = Series::new("equity", equities);
    let cash_series = Series::new("cash", cash);
    let holdings_value_series = Series::new("holdings_value", holdings_values);

    // 组合成 DataFrame
    DataFrame::new(vec![date_series, equity_series, cash_series, holdings_value_series])
}