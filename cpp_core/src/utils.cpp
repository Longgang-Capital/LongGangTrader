// cpp_core/src/utils.cpp
#include "utils.h"

#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <stdexcept>
#include <algorithm>
#include <unordered_set>

using namespace std;
using namespace arrow;

namespace utils {

// 从 Arrow 表中读取字符串类型列（如 date、symbol）
shared_ptr<StringArray> read_string_column(const shared_ptr<Table>& table, const string& col_name) {
    // 1. 检查列是否存在
    if (table->schema()->GetFieldIndex(col_name) == -1) {
        throw runtime_error("列不存在: " + col_name);
    }

    // 2. 获取列并转换为字符串类型
    auto col = table->GetColumnByName(col_name);
    if (col->type()->id() != Type::STRING) {
        throw runtime_error("列 " + col_name + " 不是字符串类型（实际类型: " + col->type()->ToString() + "）");
    }

    // 3. 转换为 StringArray 并返回
    return static_pointer_cast<StringArray>(col->chunk(0));  // 假设数据只有一个 chunk（回测场景常见）
}

// 从 Arrow 表中读取数值类型列（如 close、target_weight）
shared_ptr<DoubleArray> read_double_column(const shared_ptr<Table>& table, const string& col_name) {
    if (table->schema()->GetFieldIndex(col_name) == -1) {
        throw runtime_error("列不存在: " + col_name);
    }

    auto col = table->GetColumnByName(col_name);
    if (col->type()->id() != Type::DOUBLE) {
        throw runtime_error("列 " + col_name + " 不是数值类型（实际类型: " + col->type()->ToString() + "）");
    }

    return static_pointer_cast<DoubleArray>(col->chunk(0));
}

// 提取 Arrow 表中的所有日期，去重并按升序排序
vector<string> get_sorted_unique_dates(const shared_ptr<Table>& table, const string& date_col) {
    // 1. 读取日期列
    auto date_array = read_string_column(table, date_col);
    if (date_array->length() == 0) {
        throw runtime_error("日期列 " + date_col + " 为空");
    }

    // 2. 提取所有日期，去重
    unordered_set<string> date_set;
    for (int64_t i = 0; i < date_array->length(); ++i) {
        if (!date_array->IsNull(i)) {
            date_set.insert(date_array->GetString(i));
        }
    }

    // 3. 转换为 vector 并排序（确保回测按时间顺序执行）
    vector<string> dates(date_set.begin(), date_set.end());
    sort(dates.begin(), dates.end());  // 字符串日期可直接排序（YYYY-MM-DD 格式）

    return dates;
}

// 校验 Arrow 表是否包含必要列
void validate_required_columns(const shared_ptr<Table>& table, const vector<string>& required_cols) {
    for (const auto& col : required_cols) {
        if (table->schema()->GetFieldIndex(col) == -1) {
            throw runtime_error("缺少必要列: " + col);
        }
    }
}

}  // namespace utils