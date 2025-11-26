use pyo3::prelude::*;
use polars::prelude::*;
use pyo3_polars::{PyDataFrame};
use std::collections::HashMap;

// 声明并导入 utils 模块
mod utils;
use utils::*;
/// 从 Python 传入的回测配置参数
/// 使用 #[pyclass] 宏使其可以在 Python 中实例化
#[pyclass]
#[derive(Clone)] // Clone是必须的，以便在Rust代码中传递副本
pub struct BacktestConfig {
    #[pyo3(get, set)]
    pub initial_capital: f64,
    
    #[pyo3(get, set)]
    pub transaction_cost_pct: f64, // 明确这是百分比成本
    
    #[pyo3(get, set)]
    pub symbol_col: String,

    #[pyo3(get, set)]
    pub date_col: String,

    #[pyo3(get, set)]
    pub close_col: String,

    #[pyo3(get, set)]
    pub weight_col: String,
}

#[pymethods]
impl BacktestConfig {
    #[new] // 这个构造函数允许在 Python 中通过 `BacktestConfig()` 创建实例
    fn new(initial_capital: f64, transaction_cost_pct: f64, 
        symbol_col: String, date_col: String, close_col: String, weight_col: String) -> Self {
        BacktestConfig { initial_capital, transaction_cost_pct, symbol_col, date_col, close_col, weight_col }
    }
}
/// Rust 实现的高性能向量化回测函数
///
/// :param signals_lf: Polars LazyFrame，包含['date', 'symbol', 'target_weight']等列
/// :param market_data_lf: Polars LazyFrame，包含['date', 'symbol', 'close']等列
/// :param config: BacktestConfig 对象，包含回测参数
/// :return: Polars DataFrame，包含每日的投资组合历史记录
#[pyfunction]
fn run_vectorized_backtest_rs(
    signals_lf: PyDataFrame,
    market_data_lf: PyDataFrame,
    config: &BacktestConfig
) -> PyResult<PyDataFrame> {

    // --- 1. 数据准备和初始化 ---
    //let signals_lazy: LazyFrame = signals_lf.into();
    //let market_data_lazy: LazyFrame = market_data_lf.into();

    // 将LazyFrame转换为DataFrame进行处理（保持现有逻辑，后续可优化为纯LazyFrame操作）
    let signals: DataFrame = signals_lf.into();
    let market_data: DataFrame = market_data_lf.into();

    // 将DataFrame转换为更易于按日期查找的HashMap结构
    let market_data_map = preprocess_market_data(&market_data, config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let signals_map = preprocess_signals(&signals, config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 获取所有唯一的、排序后的交易日
    let sorted_unique_dates = get_sorted_unique_dates(&market_data, config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // --- 2. 初始化投资组合状态 (Portfolio State) ---
    let mut cash = config.initial_capital;
    let mut current_positions: PositionMap = HashMap::new(); // symbol -> shares
    let mut portfolio_history: Vec<PortfolioSnapshot> = Vec::new(); // 用于记录每日快照

    // --- 3. 按时间顺序遍历交易日 (Main Loop) ---
    for date in sorted_unique_dates {
        // 获取当日的市场价格数据，如果某天没有数据则跳过
        let market_data_for_date = match market_data_map.get(&date) {
            Some(data) => data,
            None => continue, // 跳过没有市场数据的日期
        };

        // a. 更新当前持仓市值 (Mark-to-Market)
        let holdings_value = calculate_holdings_value(&current_positions, market_data_for_date);
        let total_equity = cash + holdings_value;

        let mut turnover_rate = 0.0;

        // b. 获取当日的目标持仓权重
        if let Some(target_weights) = get_target_weights_for_date(&signals_map, &date) {
        
            // c. 计算目标持仓市值
            let target_values = calculate_target_positions_value(total_equity, target_weights);
            
            // d. 生成交易指令
            let trades = calculate_trades(&current_positions, &target_values, market_data_for_date);

            turnover_rate = calculate_turnover_rate(&trades, total_equity);

            // e. 执行交易并更新状态
            for trade in trades {
                let trade_value = trade.shares * trade.price;
                let transaction_cost = trade_value.abs() * config.transaction_cost_pct;

                // 更新现金和持仓
                cash -= trade_value + transaction_cost;
                *current_positions.entry(trade.symbol).or_insert(0.0) += trade.shares;
            }
        }    

        // f. 记录当日的投资组合快照
        portfolio_history.push(PortfolioSnapshot {
             date: date.clone(),
             equity: total_equity,
             cash,
             holdings_value,
             turnover_rate,
        });
    }

    // --- 4. 构建并返回结果 DataFrame ---
    let result_df = build_results_dataframe_from_history(&portfolio_history)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    Ok(PyDataFrame(result_df))
}


/// 定义 Python 模块
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_vectorized_backtest_rs, m)?)?;
    m.add_class::<BacktestConfig>()?;
    Ok(())
}