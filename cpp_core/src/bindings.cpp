// cpp_core/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <arrow/python/pyarrow.h>  // Arrow Python 绑定
#include "backtest.h"

namespace py = pybind11;

// 绑定 BacktestConfig 结构体（与 Rust 的 BacktestConfig 字段完全一致）
void bind_backtest_config(py::module_ &m) {
    py::class_<BacktestConfig>(m, "BacktestConfig")
        .def(py::init<double, double, std::string, std::string, std::string, std::string>(),
             py::arg("initial_capital"),
             py::arg("transaction_cost_pct"),
             py::arg("symbol_col"),
             py::arg("date_col"),
             py::arg("close_col"),
             py::arg("weight_col"))
        .def_readwrite("initial_capital", &BacktestConfig::initial_capital)
        .def_readwrite("transaction_cost_pct", &BacktestConfig::transaction_cost_pct)
        .def_readwrite("symbol_col", &BacktestConfig::symbol_col)
        .def_readwrite("date_col", &BacktestConfig::date_col)
        .def_readwrite("close_col", &BacktestConfig::close_col)
        .def_readwrite("weight_col", &BacktestConfig::weight_col);
}

// 绑定核心回测函数（函数名、参数、返回值与 Rust 完全一致）
void bind_backtest_functions(py::module_ &m) {
    m.def("run_vectorized_backtest_cpp",  // 函数名可与 Cpp 保持一致（如 run_vectorized_backtest）
          [](const py::object &signals_lf,  // 接收 Polars LazyFrame（Python 侧）
             const py::object &market_data_lf,
             const BacktestConfig &config) -> py::object {
              
              // 1. 将 Python 侧的 Polars LazyFrame 转换为 C++ Arrow Table
              //    （Polars LazyFrame 可通过 .collect() 转为 Arrow Table）
              auto signals_table = arrow::py::unwrap_table(signals_lf.ptr()).ValueOrDie();
              auto market_table = arrow::py::unwrap_table(market_data_lf.ptr()).ValueOrDie();
              
              // 2. 调用 C++ 核心回测逻辑
              BacktesterCore backtester(config);
              auto result_table = backtester.run_vectorized_backtest(signals_table, market_table);
              
              // 3. 将 Arrow Table 转换为 Python 侧的 Polars DataFrame
              return py::reinterpret_steal<py::object>(arrow::py::wrap_table(result_table));
          },
          py::arg("signals_lf"),
          py::arg("market_data_lf"),
          py::arg("config"),
          "高性能 C++ 向量化回测函数");
}

// 定义 Python 模块（模块名建议与 Rust 模块区分，如 cpp_core）
PYBIND11_MODULE(cpp_core, m) {
    arrow::py::import_pyarrow();  // 初始化 Arrow Python 绑定
    m.doc() = "C++ 高性能回测核心模块";
    
    bind_backtest_config(m);
    bind_backtest_functions(m);
}