// cpp_core/src/backtest.cpp
#include "backtest.h"
#include "utils.h"

#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/ipc/writer.h>
#include <arrow/type.h> 
#include <arrow/builder.h>
#include <arrow/status.h>

#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace std;
using namespace arrow;
using namespace utils;

// --------------------------
// BacktesterCore 构造函数
// --------------------------
BacktesterCore::BacktesterCore(const BacktestConfig& cfg) : config(cfg) {
    // 初始化回测状态：现金 = 初始资金，持仓为空
    cash = cfg.initial_capital;
    current_positions.clear();
}

// --------------------------
// 辅助函数：预处理市场数据（按日期分组）
// 输入：Arrow 表（date, symbol, close）
// 输出：unordered_map<日期, unordered_map<股票, 收盘价>>
// --------------------------
unordered_map<string, unordered_map<string, double>>
BacktesterCore::preprocess_market_data(const shared_ptr<Table>& market_data_table) {
    // 1. 校验必要列（date、symbol、close）
    vector<string> required_cols = {config.date_col, config.symbol_col, config.close_col};
    validate_required_columns(market_data_table, required_cols);

    // 2. 读取核心列
    auto date_array = read_string_column(market_data_table, config.date_col);
    auto symbol_array = read_string_column(market_data_table, config.symbol_col);
    auto close_array = read_double_column(market_data_table, config.close_col);

    // 3. 按日期分组存储数据
    unordered_map<string, unordered_map<string, double>> market_data_map;
    int64_t row_count = market_data_table->num_rows();

    for (int64_t i = 0; i < row_count; ++i) {
        // 跳过空值（回测中忽略无效数据）
        if (date_array->IsNull(i) || symbol_array->IsNull(i) || close_array->IsNull(i)) {
            continue;
        }

        string date = date_array->GetString(i);
        string symbol = symbol_array->GetString(i);
        double close = close_array->Value(i);

        // 存入哈希表：date → symbol → close
        market_data_map[date][symbol] = close;
    }

    if (market_data_map.empty()) {
        throw runtime_error("预处理后市场数据为空");
    }

    return market_data_map;
}

// --------------------------
// 辅助函数：预处理信号数据（按日期分组）
// 输入：Arrow 表（date, symbol, target_weight）
// 输出：unordered_map<日期, unordered_map<股票, 目标权重>>
// --------------------------
unordered_map<string, unordered_map<string, double>>
BacktesterCore::preprocess_signals(const shared_ptr<Table>& signals_table) {
    // 1. 校验必要列（date、symbol、target_weight）
    vector<string> required_cols = {config.date_col, config.symbol_col, config.weight_col};
    validate_required_columns(signals_table, required_cols);

    // 2. 读取核心列
    auto date_array = read_string_column(signals_table, config.date_col);
    auto symbol_array = read_string_column(signals_table, config.symbol_col);
    auto weight_array = read_double_column(signals_table, config.weight_col);

    // 3. 按日期分组存储数据
    unordered_map<string, unordered_map<string, double>> signals_map;
    int64_t row_count = signals_table->num_rows();

    for (int64_t i = 0; i < row_count; ++i) {
        if (date_array->IsNull(i) || symbol_array->IsNull(i) || weight_array->IsNull(i)) {
            continue;
        }

        string date = date_array->GetString(i);
        string symbol = symbol_array->GetString(i);
        double weight = weight_array->Value(i);

        // 存入哈希表：date → symbol → target_weight
        signals_map[date][symbol] = weight;
    }

    if (signals_map.empty()) {
        throw runtime_error("预处理后信号数据为空");
    }

    return signals_map;
}

// --------------------------
// 辅助函数：计算当前持仓市值
// --------------------------
double BacktesterCore::calculate_holdings_value(
    const unordered_map<string, double>& positions,
    const unordered_map<string, double>& market_data_for_date
) {
    double total_value = 0.0;
    for (const auto& [symbol, quantity] : positions) {
        // 查找当日股票收盘价（若没有则视为 0，忽略该持仓）
        auto close_it = market_data_for_date.find(symbol);
        if (close_it != market_data_for_date.end()) {
            total_value += quantity * close_it->second;
        }
    }
    return total_value;
}

// --------------------------
// 辅助函数：构建回测结果 Arrow 表（用于返回给 Python）
// --------------------------
shared_ptr<Table> BacktesterCore::build_results_table(const vector<PortfolioSnapshot>& history) {
    // 1. 定义结果表的列结构（与 Python 侧 DataFrame 列对齐）
    vector<shared_ptr<Field>> fields = {
        field("date", utf8()),
        field("equity", float64()),
        field("cash", float64()),
        field("holdings_value", float64()),
        field("turnover_rate", float64())
    };
    shared_ptr<arrow::Schema> schema = arrow::schema(fields);

    // 2. 创建 RecordBatchBuilder（批量构建 Arrow 数据）
    auto memory_pool = default_memory_pool();
    // TODO 
    arrow::StringBuilder date_builder(memory_pool);
    arrow::DoubleBuilder equity_builder(memory_pool);
    arrow::DoubleBuilder cash_builder(memory_pool);
    arrow::DoubleBuilder holdings_value_builder(memory_pool);
    arrow::DoubleBuilder turnover_rate_builder(memory_pool);

    // 3. 遍历历史数据，填充各列
    for (const auto& snapshot : history) {
        if (!date_builder.Append(snapshot.date).ok()) {
            throw runtime_error("构建 date 列失败");
        }
        if (!equity_builder.Append(snapshot.equity).ok()) {
            throw runtime_error("构建 equity 列失败");
        }
        if (!cash_builder.Append(snapshot.cash).ok()) {
            throw runtime_error("构建 cash 列失败");
        }
        if (!holdings_value_builder.Append(snapshot.holdings_value).ok()) {
            throw runtime_error("构建 holdings_value 列失败");
        }
        if (!turnover_rate_builder.Append(snapshot.turnover_rate).ok()) {
            throw runtime_error("构建 turnover_rate 列失败");
        }
    }

    // 4. Finish 各列，生成 Array
    shared_ptr<Array> date_array, equity_array, cash_array, holdings_value_array, turnover_rate_array;
    
    if (!date_builder.Finish(&date_array).ok()) {
        throw runtime_error("完成 date 数组失败");
    }
    if (!equity_builder.Finish(&equity_array).ok()) {
        throw runtime_error("完成 equity 数组失败");
    }
    if (!cash_builder.Finish(&cash_array).ok()) {
        throw runtime_error("完成 cash 数组失败");
    }
    if (!holdings_value_builder.Finish(&holdings_value_array).ok()) {
        throw runtime_error("完成 holdings_value 数组失败");
    }
    if (!turnover_rate_builder.Finish(&turnover_rate_array).ok()) {
        throw runtime_error("完成 turnover_rate 数组失败");
    }

    // 5. 构建并返回 Table
    return Table::Make(schema, {
        date_array,
        equity_array,
        cash_array,
        holdings_value_array,
        turnover_rate_array
    });
    // TODO
}

// --------------------------
// 核心回测函数（对外暴露的核心接口）
// --------------------------
shared_ptr<Table> BacktesterCore::run_vectorized_backtest(
    const shared_ptr<Table>& signals_table,
    const shared_ptr<Table>& market_data_table
) {
    try {
        // 1. 预处理数据（市场数据 + 信号数据）
        cout << "开始预处理数据..." << endl;
        auto market_data_map = preprocess_market_data(market_data_table);
        auto signals_map = preprocess_signals(signals_table);

        // 2. 获取所有回测日期（取市场数据和信号数据的交集，按升序排序）
        auto market_dates = get_sorted_unique_dates(market_data_table, config.date_col);
        auto signal_dates = get_sorted_unique_dates(signals_table, config.date_col);
        vector<string> backtest_dates;
        set_intersection(
            market_dates.begin(), market_dates.end(),
            signal_dates.begin(), signal_dates.end(),
            back_inserter(backtest_dates)
        );

        if (backtest_dates.empty()) {
            throw runtime_error("市场数据和信号数据无共同日期，无法回测");
        }
        cout << "回测日期范围: " << backtest_dates.front() << " 至 " << backtest_dates.back() << endl;

        // 3. 初始化回测状态和结果历史
        vector<PortfolioSnapshot> portfolio_history;
        double prev_holdings_value = 0.0;  // 上一日持仓市值（用于计算换手率）

        // 4. 回测主循环（按日期遍历）
        for (const auto& date : backtest_dates) {
            // 4.1 获取当日市场数据和信号数据
            auto& market_data_today = market_data_map[date];
            auto& signals_today = signals_map[date];

            // 4.2 计算当日总资产（现金 + 持仓市值）
            double holdings_value_today = calculate_holdings_value(current_positions, market_data_today);
            double total_equity = cash + holdings_value_today;

            // 4.3 计算目标持仓（按信号权重分配资金）
            unordered_map<string, double> target_positions;  // 目标持仓（股数）
            double total_weight = 0.0;

            // 先计算总权重（用于归一化，避免权重和不为 1 的情况）
            for (const auto& [symbol, weight] : signals_today) {
                total_weight += weight;
            }

            // 按归一化后的权重分配资金，计算目标股数
            for (const auto& [symbol, weight] : signals_today) {
                if (total_weight <= 0) break;

                // 目标资金占比 = 个股权重 / 总权重
                double target_weight = weight / total_weight;
                double target_cash = total_equity * target_weight;

                // 目标股数 = 目标资金 / 当日收盘价（向下取整，避免碎股）
                auto close_it = market_data_today.find(symbol);
                if (close_it != market_data_today.end() && close_it->second > 0) {
                    double target_quantity = floor(target_cash / close_it->second);
                    target_positions[symbol] = target_quantity;
                }
            }

            // 4.4 生成交易指令并执行（调仓）
            double turnover = 0.0;  // 当日成交额（用于计算换手率）
            for (const auto& [symbol, target_qty] : target_positions) {
                // 当前持仓股数（无则为 0）
                double current_qty = current_positions.count(symbol) ? current_positions[symbol] : 0.0;
                double trade_qty = target_qty - current_qty;  // 交易股数（正=买入，负=卖出）

                if (trade_qty == 0) continue;  // 无调仓需求

                // 获取当日收盘价（确保存在）
                auto close_it = market_data_today.find(symbol);
                if (close_it == market_data_today.end()) continue;
                double close_price = close_it->second;

                // 计算交易金额（含手续费）
                double trade_amount = abs(trade_qty) * close_price;
                double fee = trade_amount * config.transaction_cost_pct;  // 交易成本
                turnover += trade_amount;  // 累计当日成交额

                // 更新现金和持仓
                if (trade_qty > 0) {
                    // 买入：现金减少（交易金额 + 手续费）
                    if (cash < trade_amount + fee) {
                        continue;  // 现金不足，跳过该笔交易
                    }
                    cash -= (trade_amount + fee);
                    current_positions[symbol] = target_qty;
                } else {
                    // 卖出：现金增加（交易金额 - 手续费）
                    cash += (trade_amount - fee);
                    current_positions[symbol] = target_qty;
                    if (target_qty <= 0) {
                        current_positions.erase(symbol);  // 清仓后移除该股票
                    }
                }
            }

            // 4.5 清仓无信号的股票（信号中没有的股票，全部卖出）
            vector<string> symbols_to_sell;
            for (const auto& [symbol, current_qty] : current_positions) {
                if (signals_today.count(symbol) == 0) {
                    symbols_to_sell.push_back(symbol);
                }
            }

            for (const auto& symbol : symbols_to_sell) {
                auto close_it = market_data_today.find(symbol);
                if (close_it == market_data_today.end()) continue;

                double current_qty = current_positions[symbol];
                double trade_amount = current_qty * close_it->second;
                double fee = trade_amount * config.transaction_cost_pct;

                // 卖出后更新现金和持仓
                cash += (trade_amount - fee);
                turnover += trade_amount;
                current_positions.erase(symbol);
            }

            // 4.6 计算换手率（当日成交额 / 上一日总资产）
            double turnover_rate = 0.0;
            if (prev_holdings_value > 0) {
                turnover_rate = turnover / prev_holdings_value;
            }

            // 4.7 记录当日快照
            portfolio_history.emplace_back(PortfolioSnapshot{
                date,
                total_equity,
                cash,
                holdings_value_today,
                turnover_rate
            });

            // 4.8 更新上一日持仓市值（用于下一日换手率计算）
            prev_holdings_value = holdings_value_today;
        }

        // 5. 构建回测结果 Arrow 表（返回给 Python）
        cout << "回测完成，共 " << portfolio_history.size() << " 个交易日" << endl;
        return build_results_table(portfolio_history);

    } catch (const exception& e) {
        // 捕获异常并重新抛出（Python 侧可捕获）
        throw runtime_error("回测失败: " + string(e.what()));
    }
}