use pyo3::prelude::*;
use polars::prelude::*;
use pyo3_polars::{PyDataFrame};

/// 在 Rust 中执行高性能回测的函数
/// 它接收 Python 中的 DataFrame 作为输入，并返回一个包含结果的 DataFrame
#[pyfunction]
fn run_backtest_rs(signals_df: PyDataFrame, market_data_df: PyDataFrame) -> PyResult<PyDataFrame> {
    // PyDataFrame 自动转换为 Polars DataFrame
    let signals: DataFrame = signals_df.into();
    let market_data: DataFrame = market_data_df.into();

    println!("Hello from Rust! Received data from Python via Polars.");
    println!("Signals DataFrame shape: {:?}", signals.shape());
    println!("Market Data DataFrame shape: {:?}", market_data.shape());

    // 在这里实现您的高性能回测逻辑...
    // 这是一个示例，返回一个模拟的结果 DataFrame
    let dates = Series::new("date", &["2023-01-01", "2023-01-02", "2023-01-03"]);
    let equity = Series::new("equity", &[1000000.0, 1000500.0, 1001000.0]);
    
    let result_df = DataFrame::new(vec![dates, equity])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 将 Polars DataFrame 转换回 PyDataFrame 以返回给 Python
    Ok(PyDataFrame(result_df))
}

/// 定义 Python 模块
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest_rs, m)?)?;
    Ok(())
}
