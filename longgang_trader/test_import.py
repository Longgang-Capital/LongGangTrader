import os
import sys

if os.name == 'nt': # Windows
    os.system('chcp 65001 > nul')

# 2. 导入你的模块
import cpp_core

# 3. 打印测试
print("成功导入！模块内容如下：")
print(dir(cpp_core))

# 测试
import polars as pl
import pandas as pd
from backtesting.backtester_cpp import Backtester, BaseStrategy

# 创建测试策略
class TestStrategy(BaseStrategy):
    def generate_signals_for_all_dates(self):
        dates = ['2023-01-01', '2023-01-02', '2023-01-03']
        symbols = ['AAPL', 'GOOGL']
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'target_weight': 0.5
                })
        return pd.DataFrame(data)

# 创建市场数据
dates = ['2023-01-01', '2023-01-02', '2023-01-03']
symbols = ['AAPL', 'GOOGL']
data = []
for date in dates:
    for symbol in symbols:
        data.append({
            'date': date,
            'symbol': symbol,
            'close': 100.0
        })

market_data = pd.DataFrame(data)

# 运行回测
strategy = TestStrategy(None)
config = {
    'initial_capital': 100000,
    'transaction_cost': 0.001,
    'symbol_col': 'symbol',
    'date_col': 'date',
    'close_col': 'close'
}

backtester = Backtester(strategy, market_data, config)
backtester.run_backtest_cpp()

if backtester.portfolio_history is not None:
    print('回测成功完成!')
    print(backtester.portfolio_history.head())
else:
    print('回测失败')





