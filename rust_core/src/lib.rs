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

    #[pyo3(get, set)]
    pub volume_col: String,

    #[pyo3(get, set)]
    pub preclose_col: String, // 前收盘价列名

    #[pyo3(get, set)]
    pub limit_pct: f64, // 涨跌停比例，如0.1表示10%

    #[pyo3(get, set)]
    pub rebalance_days: i32, // 调仓天数，1表示每日调仓，N表示每N天调仓
}

#[pymethods]
impl BacktestConfig {
    #[new] // 这个构造函数允许在 Python 中通过 `BacktestConfig()` 创建实例
    #[pyo3(signature = (initial_capital, transaction_cost_pct, 
        symbol_col, date_col, close_col, weight_col, volume_col, preclose_col, 
        limit_pct=0.1, rebalance_days=1))]
    fn new(
        initial_capital: f64,
        transaction_cost_pct: f64,
        symbol_col: String,
        date_col: String,
        close_col: String,
        weight_col: String,
        volume_col: String,
        preclose_col: String,
        limit_pct: f64,
        rebalance_days: i32,
    ) -> Self {
        BacktestConfig {
            initial_capital,
            transaction_cost_pct,
            symbol_col,
            date_col,
            close_col,
            weight_col,
            volume_col,
            preclose_col,
            limit_pct,
            rebalance_days,
        }
    }
}
/// Rust 实现的高性能向量化回测函数
///
/// :param signals_lf: Polars LazyFrame，包含['date', 'symbol', 'target_weight']等列
/// :param market_data_path: 市场数据的文件路径 (parquet, csv, bin, pkl)
/// :param config: BacktestConfig 对象，包含回测参数
/// :return: Polars DataFrame，包含每日的投资组合历史记录
#[pyfunction]
fn run_vectorized_backtest_rs(
    signals_lf: PyDataFrame,
    market_data_path: String,
    config: &BacktestConfig
) -> PyResult<PyDataFrame> {

    // --- 1. 数据准备和初始化 ---
    let market_data: DataFrame = load_market_data(&market_data_path)?;

    let signals: DataFrame = signals_lf.into();

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

    // 初始化交易日计数器
    let mut day_counter = 0;
    // --- 新增：价格记忆缓存 ---
    let mut last_known_prices: HashMap<String, f64> = HashMap::new();

    // --- 3. 按时间顺序遍历交易日 (Main Loop) ---
    for date in sorted_unique_dates {
        // 获取当日的市场价格数据，如果某天没有数据则跳过
        let market_data_for_date = match market_data_map.get(&date) {
            Some(data) => data,
            None => continue, // 跳过没有市场数据的日期
        };
        // 遍历当天所有有行情的股票，更新它们的最新价格
        for (symbol, entry) in market_data_for_date.iter() {
            if entry.price > 0.0 {
                last_known_prices.insert(symbol.clone(), entry.price);
            }
        }

        // a. 更新当前持仓市值 (Mark-to-Market)
        //let holdings_value = calculate_holdings_value(&current_positions, market_data_for_date);
        //let total_equity = cash + holdings_value;

        let mut turnover_rate = 0.0;

        // b. 检查是否需要调仓（每N天调一次）
        let should_rebalance = day_counter % config.rebalance_days == 0;

        // c. 获取当日的目标持仓权重（如果有信号且需要调仓）
        if should_rebalance {
            if let Some(target_weights) = get_target_weights_for_date(
                &signals_map, &date) {

                // 1. 计算交易前权益 (用于确定买多少)
                // 注意：这里用的是当前持仓在当前价格下的市值 + 现金
                let pre_trade_holdings_value = calculate_holdings_value(
                    &current_positions, market_data_for_date,&last_known_prices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                let pre_trade_equity = cash + pre_trade_holdings_value;
                // d. 计算目标持仓市值
                let target_values = calculate_target_positions_value(
                    pre_trade_equity, target_weights);
                // e. 生成交易指令（包含涨跌停限制）
                let trades = calculate_trades(
                    &current_positions, &target_values, market_data_for_date, config.limit_pct);

                turnover_rate = calculate_turnover_rate(&trades, pre_trade_equity);

                // f. 执行交易并更新状态
                for trade in trades {
                    let trade_value = trade.shares * trade.price;
                    let transaction_cost = trade_value.abs() * config.transaction_cost_pct;

                    // 更新现金和持仓
                    cash -= trade_value + transaction_cost;
                    *current_positions.entry(trade.symbol).or_insert(0.0) += trade.shares;
                }
            }
        }
        // --- 修正点 2: 会计核算 ---
        // 交易完成后，重新计算持仓市值和总权益
        // 这样 equity 才会反映扣除手续费后的真实净值，且与 cash 对应
        let final_holdings_value = calculate_holdings_value(
            &current_positions, market_data_for_date, &last_known_prices)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let final_equity = cash + final_holdings_value;

        // g. 记录当日的投资组合快照
        portfolio_history.push(PortfolioSnapshot {
             date: date.clone(),
             equity: final_equity,
             cash,
             holdings_value: final_holdings_value,
             turnover_rate,
        });

        // h. 更新交易日计数器
        day_counter += 1;
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