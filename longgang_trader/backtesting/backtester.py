import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 全局设置中文显示
plt.rcParams["font.family"] = ["SimHei","Microsoft YaHei"] # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 尝试导入由 PyO3 构建的 Rust 核心模块
# 在运行前，需要先在项目根目录执行 `maturin develop`
try:
    # 导入 Rust 模块中定义的函数和类
    from rust_core import run_vectorized_backtest_rs, BacktestConfig # type: ignore
except ImportError:
    print("错误: 无法导入 'rust_core' 模块。")
    print("请确保您已经在项目根目录下运行了 'maturin develop' 来编译和安装 Rust 核心。")
    # 如果导入失败，将它们设置为 None，以便在后续代码中进行检查
    run_vectorized_backtest_rs = None
    BacktestConfig = None

class Backtester:
    """
    回测引擎，用于评估策略的历史表现。
    """
    def __init__(self, strategy, data_path, config:dict):
        """
        :param strategy: 交易策略对象。
        :param data_path: 行情数据的文件路径 (str)。
        :param initial_capital: 初始资金。
        :param transaction_cost: 交易成本（手续费、滑点等）。
        """
        self.strategy = strategy
        self.data_path = data_path
        self.config = config
        self.initial_capital = config.get("initial_capital", 1_000_000)
        self.transaction_cost = config.get("transaction_cost", 0.001)  # 0.1%
        self.symbol_col = config.get("symbol_col", "symbol")
        self.date_col = config.get("date_col", "date")
        self.close_col = config.get("close_col", "close")
        self.weight_col = config.get("weight_col", "target_weight")

        self.portfolio_history = None

    def _run_backtest_internal(self, signals):
        """
        内部回测逻辑，接收一个信号DataFrame并运行回测。
        """
        if run_vectorized_backtest_rs is None:
            return None

        if BacktestConfig is None:
            print("错误: 'BacktestConfig' 未定义，无法运行回测。")
            return None

        config = BacktestConfig(
            initial_capital=self.initial_capital,
            transaction_cost_pct=self.transaction_cost,
            symbol_col=self.symbol_col,
            date_col=self.date_col,
            close_col=self.close_col,
            weight_col=self.weight_col
        )
        signals_pl = pl.from_pandas(signals)

        try:
            result_pl = run_vectorized_backtest_rs(signals_pl, self.data_path, config)
            return result_pl.to_pandas()
        except Exception as e:
            print(f"错误: 调用 Rust 回测函数时发生异常: {e}")
            return None

    def run_backtest(self):
        """
        执行回测。
        通过直接调用 Rust 函数来运行高性能的回测计算。
        """
        signals = self.strategy.generate_signals_for_all_dates()
        self.portfolio_history = self._run_backtest_internal(signals)
        if self.portfolio_history is not None:
            print("Rust 回测成功完成。")

    def get_portfolio_history(self):
        """
        获取投资组合的每日市值、持仓等历史记录。
        """
        return self.portfolio_history

    def calculate_performance_metrics(self, portfolio_history=None):
        """
        计算回测的各项性能指标。
        :param portfolio_history: (可选) 要计算指标的投资组合历史DataFrame。如果为None，则使用self.portfolio_history。
        :return: 包含年化收益、夏普比率、最大回撤等指标的字典。
        """
        if portfolio_history is None:
            portfolio_history = self.portfolio_history

        if portfolio_history is None:
            print("没有可用的投资组合历史数据。")
            return None

        df = portfolio_history.copy()
        try:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        except KeyError:
            print(f"错误：投资组合中缺少日期列 '{self.date_col}' ")
            return None

        df['strategy_net_value'] = df['equity'] / self.initial_capital

        # 年化收益率
        total_return = df['strategy_net_value'].iloc[-1] - 1.0
        days = (df[self.date_col].iloc[-1] - df[self.date_col].iloc[0]).days
        if days == 0:
            annualized_return = 0.0
        else:
            annualized_return = (1 + total_return) ** (365.0 / days) - 1
        
        # 最大回撤
        df['cum_max'] = df['strategy_net_value'].cummax()
        df['drawdown'] = df['strategy_net_value'] / df['cum_max'] - 1
        max_drawdown = df['drawdown'].min()

        # 夏普比率
        daily_returns = df['strategy_net_value'].pct_change().dropna()
        annualized_std = daily_returns.std() * np.sqrt(252)
        risk_free_rate = 0.015
        if days == 0:
            rf = 0.0
        else:
            rf = (1 + risk_free_rate) ** (days / 365.0) - 1
        excess_return_annualized = annualized_return - rf
        sharpe_ratio = excess_return_annualized / annualized_std if annualized_std != 0 else np.nan
        
        return {
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

    def plot_equity_curve(self, portfolio_history=None, title="Equity Curve", save_path=None):
        """
        绘制资金曲线。
        :param portfolio_history: (可选) 要绘制的投资组合历史DataFrame。如果为None，则使用self.portfolio_history。
        :param title: (可选) 图表标题。
        :param save_path: (可选) 图片保存路径。如果为None，则不保存图片。
        """
        if portfolio_history is None:
            portfolio_history = self.portfolio_history

        if portfolio_history is None:
            print("没有可用的投资组合历史数据。")
            return

        df = portfolio_history.copy()
        try:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        except KeyError:
            print(f"错误：投资组合中缺少日期列 '{self.date_col}' ")
            return

        df['strategy_net_value'] = df['equity'] / self.initial_capital

        # 计算性能指标
        metrics = self.calculate_performance_metrics(portfolio_history)
        if metrics is None:
            return

        # 绘制资金曲线
        plt.figure(figsize=(12,8))
        ax = plt.subplot(1,1,1)
        ax.plot(df[self.date_col], df['strategy_net_value'], label='策略净值', color='#2ECC71',linewidth=2)
        #美化图表
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('日期',fontsize=14,labelpad=10)
        #设置x轴刻度、标签、输出格式
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, fontsize=12)

        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        #在图表上显示指标
        stats_text = (f"年化收益率: {metrics['annualized_return']:.2%}\n"
                    f"最大回撤: {metrics['max_drawdown']:.2%}\n"
                    f"夏普比率: {metrics['sharpe_ratio']:.2f}")

        ax.text(0.01,0.90,stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()

        # 保存图片或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
            plt.close()
        else:
            plt.show()

    def run_grouped_backtest(self, save_plots=True, results_dir=None):
        """
        为每个分组分别运行回测，并返回每个组的回测结果和性能指标。
        :param save_plots: 是否保存资金曲线图片
        :param results_dir: 结果保存目录，如果为None则使用默认目录
        """
        all_signals = self.strategy.generate_signals_for_all_dates()

        group_col = self.strategy.group_col
        if group_col not in all_signals.columns:
            print(f"错误: 信号中缺少分组列 '{group_col}'。")
            return None

        groups = all_signals[group_col].unique()
        all_results = {}

        for group in sorted(groups):
            print(f"\n----- 正在回测分组: {group} -----")
            group_signals = all_signals[all_signals[group_col] == group]

            portfolio_history = self._run_backtest_internal(group_signals)

            if portfolio_history is not None:
                metrics = self.calculate_performance_metrics(portfolio_history)
                all_results[group] = {
                    'portfolio_history': portfolio_history,
                    'metrics': metrics
                }
                print(f"分组 {group} 回测完成。")

                # 绘制并保存资金曲线
                if save_plots:
                    if results_dir:
                        import os
                        os.makedirs(results_dir, exist_ok=True)
                        plot_path = os.path.join(results_dir, f'equity_curve_group_{group}.png')
                    else:
                        plot_path = None

                    self.plot_equity_curve(
                        portfolio_history,
                        title=f"Group {group} Equity Curve",
                        save_path=plot_path
                    )
                else:
                    self.plot_equity_curve(portfolio_history, title=f"Group {group} Equity Curve")
            else:
                print(f"分组 {group} 回测失败。")

        return all_results


class BaseStrategy:
    """
    交易策略基类。
    """
    def __init__(self, factor_data,config:dict={}):
        self.factor_data = factor_data
        self.config = config
        self.date_col = config.get("date_col", "date")
        self.symbol_col = config.get("symbol_col", "order_book_id")
        self.factor_value_col = config.get("factor_value_col", "factor_value")
        self.weight_col = config.get("weight_col", "target_weight")

    def generate_signals(self, current_date, current_positions):
        """
        根据因子值生成交易信号（目标持仓）。
        :param current_date: 当前交易日。
        :param current_positions: 当前持仓。
        :return: 目标持仓权重 (dict)。
        """
        raise NotImplementedError

    def generate_signals_for_all_dates(self):
        """
        (可选) 为所有日期一次性生成信号，用于传递给高性能回测器。
        """
        # 这是一个示例实现，您需要根据策略逻辑进行调整
        # 遍历所有交易日，调用 generate_signals
        pass


class SimpleLayeredStrategy(BaseStrategy):
    """一个简单的策略，在每个交易日，等权重买入因子值最高的一只股票"""
    def generate_signals_for_all_dates(self):
        # 合并因子数据，方便处理
        data = self.factor_data.copy()

        # 在每个日期，找到因子值最高的股票
        top_stocks = data.loc[data.groupby(self.date_col)[self.factor_value_col].idxmax()]

        # 设置目标权重为1.0
        top_stocks[self.weight_col] = 1.0

        signals = top_stocks[[self.date_col, self.symbol_col, self.weight_col]]
        #signals.columns = ['date', 'symbol', 'target_weight']

        print("\n生成的交易信号 (signals):")
        print(signals)
        return signals


class TopNFactorStrategy(BaseStrategy):
    """
    投资因子值前N的交易策略。
    在每个交易日，选择因子值最高的N只股票进行等权重投资。
    """
    def __init__(self, factor_data, config:dict={}):
        super().__init__(factor_data, config)
        # 获取配置参数，默认为前5只股票
        self.top_n = config.get("top_n", 5)

    def generate_signals_for_all_dates(self):
        """
        为所有日期一次性生成交易信号。
        在每个交易日，选择因子值最高的N只股票，进行等权重分配。
        """
        # 复制因子数据以避免修改原始数据
        data = self.factor_data.copy()

        # 按日期分组，在每个日期内按因子值降序排列，取前N只股票
        def get_top_n_stocks(group):
            return group.nlargest(self.top_n, self.factor_value_col)

        top_stocks = data.groupby(self.date_col).apply(get_top_n_stocks).reset_index(drop=True)

        # 计算等权重：每只股票分配 1/N 的权重
        top_stocks[self.weight_col] = 1.0 / self.top_n

        # 选择需要的列
        signals = top_stocks[[self.date_col, self.symbol_col, self.weight_col]]

        print(f"\n生成的交易信号 (Top {self.top_n} Factor Strategy):")
        print(f"总交易日数: {signals[self.date_col].nunique()}")
        print(f"总信号数: {len(signals)}")
        print(f"平均每个交易日选择的股票数: {len(signals) / signals[self.date_col].nunique():.2f}")
        print("信号数据预览:")
        print(signals.head())

        return signals

    def generate_signals(self, current_date, current_positions):
        """
        根据当前日期和持仓生成交易信号。
        :param current_date: 当前交易日
        :param current_positions: 当前持仓
        :return: 目标持仓权重 (dict)
        """
        # 获取当前日期的因子数据
        current_data = self.factor_data[self.factor_data[self.date_col] == current_date]

        if current_data.empty:
            return {}

        # 按因子值降序排列，取前N只股票
        top_stocks = current_data.nlargest(self.top_n, self.factor_value_col)

        # 计算等权重
        target_weights = {}
        weight_per_stock = 1.0 / len(top_stocks) if len(top_stocks) > 0 else 0

        for _, row in top_stocks.iterrows():
            symbol = row[self.symbol_col]
            target_weights[symbol] = weight_per_stock

        return target_weights


class GroupedFactorStrategy(BaseStrategy):
    """
    一个简单的策略，在每个交易日，使用优化权重买入各分组内的股票。
    该策略假设优化器已经计算好了每个股票的目标权重。
    """
    def __init__(self, factor_data, config:dict={}):
        super().__init__(factor_data, config)
        self.group_col = config.get("group_col", "group")
        self.optimized_weight_col = config.get("optimized_weight_col", "optimized_weight")


    def generate_signals_for_all_dates(self):
        # The optimizer has already calculated weights, so this strategy just formats the data
        signals = self.factor_data.copy()
        signals = signals.rename(columns={self.optimized_weight_col: self.weight_col})
        return signals
