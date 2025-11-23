// cpp_core/include/utils.h
#ifndef UTILS_H
#define UTILS_H

#include <arrow/table.h>
#include <vector>
#include <string>
#include <memory>

using namespace std;
using namespace arrow;

namespace utils {

// 读取字符串类型列（date、symbol 等）
shared_ptr<StringArray> read_string_column(const shared_ptr<Table>& table, const string& col_name);

// 读取数值类型列（close、target_weight 等）
shared_ptr<DoubleArray> read_double_column(const shared_ptr<Table>& table, const string& col_name);

// 提取并排序所有唯一日期
vector<string> get_sorted_unique_dates(const shared_ptr<Table>& table, const string& date_col);

// 校验表是否包含必要列
void validate_required_columns(const shared_ptr<Table>& table, const vector<string>& required_cols);

}  // namespace utils

#endif // UTILS_H