// src/utils.rs

use std::collections::HashMap;
use polars::prelude::*;

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
        .map(|(symbol, &weight)| (symbol.clone(), total_equity * weight))
        .collect()
}

/// 比较当前持仓和目标持仓，生成具体的交易指令列表
pub fn calculate_trades(
    current_positions: &PositionMap,
    target_positions_value: &HashMap<String, f64>,
    market_data_for_date: &DailyMarketData,
) -> Vec<Trade> {
    // 这是一个简化的实现，实际中需要处理各种情况
    let mut trades = Vec::new();
    // ... 比较和计算逻辑 ...
    trades
}


/// 将记录的投资组合历史快照列表转换为 Polars DataFrame
pub fn build_results_dataframe_from_history(
    portfolio_history: &[PortfolioSnapshot],
) -> PolarsResult<DataFrame> {
    // 从快照中提取列数据
    let dates: Vec<&str> = portfolio_history.iter().map(|s| s.date.as_str()).collect();
    let equities: Vec<f64> = portfolio_history.iter().map(|s| s.equity).collect();
    let cash: Vec<f64> = portfolio_history.iter().map(|s| s.cash).collect();
    let holdings_values: Vec<f64> = portfolio_history.iter().map(|s| s.holdings_value).collect();

    // 创建 Polars Series
    let date_series = Series::new("date", dates);
    let equity_series = Series::new("equity", equities);
    let cash_series = Series::new("cash", cash);
    let holdings_value_series = Series::new("holdings_value", holdings_values);

    // 组合成 DataFrame
    DataFrame::new(vec![date_series, equity_series, cash_series, holdings_value_series])
}