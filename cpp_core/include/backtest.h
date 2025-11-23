// cpp_core/include/backtest.h
#ifndef BACKTEST_H
#define BACKTEST_H

#include <string>
#include <unordered_map>
#include <vector>
#include <arrow/api.h>  // 依赖 Apache Arrow 处理列存数据（类似 Rust 的 polars）

// 回测配置结构体（对应 Rust 的 BacktestConfig）
struct BacktestConfig {
    double initial_capital;          // 初始资金
    double transaction_cost_pct;     // 交易成本百分比（如 0.001 表示 0.1%）
    std::string symbol_col;          // 股票代码列名
    std::string date_col;            // 日期列名
    std::string close_col;           // 收盘价列名
    std::string weight_col;          // 目标权重列名
};

// 每日持仓快照（用于记录回测结果）
struct PortfolioSnapshot {
    std::string date;                // 日期
    double equity;                   // 总资产
    double cash;                     // 现金
    double holdings_value;           // 持仓市值
    double turnover_rate;            // 换手率
};

// 回测核心类
class BacktesterCore {
private:
    BacktestConfig config;
    // 内部状态：当前现金、持仓（股票代码 -> 持股数量）
    double cash;
    std::unordered_map<std::string, double> current_positions;

public:
    // 构造函数：通过配置初始化
    explicit BacktesterCore(const BacktestConfig& cfg);

    // 核心回测函数：接收 Arrow 格式的信号数据和市场数据，返回回测结果
    std::shared_ptr<arrow::Table> run_vectorized_backtest(
        const std::shared_ptr<arrow::Table>& signals_table,
        const std::shared_ptr<arrow::Table>& market_data_table
    );

private:
    // 辅助函数：预处理市场数据为按日期索引的结构
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
    preprocess_market_data(const std::shared_ptr<arrow::Table>& market_data_table);

    // 辅助函数：预处理信号数据为按日期索引的结构
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
    preprocess_signals(const std::shared_ptr<arrow::Table>& signals_table);

    // 辅助函数：计算持仓市值
    double calculate_holdings_value(
        const std::unordered_map<std::string, double>& positions,
        const std::unordered_map<std::string, double>& market_data_for_date
    );

    // 辅助函数：将回测结果转换为 Arrow Table（方便 Python 读取）
    std::shared_ptr<arrow::Table> build_results_table(
        const std::vector<PortfolioSnapshot>& history
    );
};

#endif // BACKTEST_H