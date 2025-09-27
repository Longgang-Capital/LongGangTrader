use pyo3::prelude::*;
use polars::prelude::*;
use pyo3_polars::{PyDataFrame};
use std::collections::HashMap;
// 新增: 声明 utils 模块，这会告诉 Rust 编译器去查找 src/utils.rs 文件
mod utils;
// 新增: 导入 utils 模块中的所有公共项，方便直接使用
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
}

#[pymethods]
impl BacktestConfig {
    #[new] // 这个构造函数允许在 Python 中通过 `BacktestConfig()` 创建实例
    fn new(initial_capital: f64, transaction_cost_pct: f64) -> Self {
        BacktestConfig { initial_capital, transaction_cost_pct }
    }
}
/// Rust 实现的高性能向量化回测函数
/// 
/// :param signals_df: Polars DataFrame，包含['date', 'symbol', 'target_weight']等列
/// :param market_data_df: Polars DataFrame，包含['date', 'symbol', 'close']等列
/// :param config: BacktestConfig 对象，包含回测参数
/// :return: Polars DataFrame，包含每日的投资组合历史记录
#[pyfunction]
fn run_vectorized_backtest_rs(
    signals_df: PyDataFrame, 
    market_data_df: PyDataFrame,
    config: &BacktestConfig
) -> PyResult<PyDataFrame> {

    // --- 1. 数据准备和初始化 ---
    let signals: DataFrame = signals_df.into();
    let market_data: DataFrame = market_data_df.into();
    // 对数据进行预处理，例如按日期排序，构建方便查询的数据结构 (e.g., HashMap<Date, HashMap<Symbol, Price>>)

    // --- 2. 初始化投资组合状态 (Portfolio State) ---
    let mut cash = config.initial_capital;
    let mut total_equity = config.initial_capital;
    let mut current_positions: HashMap<String, f64> = HashMap::new(); // symbol -> shares
    let mut portfolio_history: Vec<PortfolioSnapshot> = Vec::new(); // 用于记录每日快照
    let sorted_unique_dates = vec!["2023-01-01".to_string(), "2023-01-02".to_string()]; // 假设已获取

    // --- 3. 按时间顺序遍历交易日 (Main Loop) ---
    for date in sorted_unique_dates {
        // a. 更新当前持仓市值 (Mark-to-Market)
        let holdings_value = calculate_holdings_value(&current_positions, &HashMap::new() /* 示例数据 */);
        total_equity = cash + holdings_value;

        // b. 获取当日的目标持仓权重
        if let Some(target_weights) = get_target_weights_for_date(&HashMap::new() /* 示例数据 */, &date) {
        
            // c. 计算目标持仓 (调用 utils 函数)
            let target_values = calculate_target_positions_value(total_equity, target_weights);
            
            // d. 生成交易 (调用 utils 函数)
            let trades = calculate_trades(&current_positions, &target_values, &HashMap::new());

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
             date: date.to_string(),
             equity: total_equity,
             cash,
             holdings_value,
             turnover_rate: 0.0, // 这里可以根据实际情况计算
        //     // ... 其他需要记录的指标
        });
    }

    // --- 4. 构建并返回结果 DataFrame ---
    // let result_df = build_results_dataframe_from_history(&portfolio_history);
    // Ok(PyDataFrame(result_df))
    
    // 这是一个示例，返回一个模拟的结果 DataFrame
    let dates = Series::new("date", &["2023-01-01", "2023-01-02", "2023-01-03"]);
    let equity = Series::new("equity", &[config.initial_capital, 1000500.0, 1001000.0]);
    let result_df = DataFrame::new(vec![dates, equity]).unwrap();

    Ok(PyDataFrame(result_df))
}


/// 定义 Python 模块
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_vectorized_backtest_rs, m)?)?;
    m.add_class::<BacktestConfig>()?;
    Ok(())
}