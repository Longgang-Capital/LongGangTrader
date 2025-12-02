import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from longgang_trader.backtesting.strategy import BaseStrategy
from typing import Union

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
    def __init__(self, strategy: BaseStrategy, data_path: str, config:dict):
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
        self.symbol_col = strategy.symbol_col
        self.date_col = strategy.date_col
        self.close_col = config.get("close_col", "close")
        self.weight_col = strategy.weight_col
        self.volume_col = config.get("volume_col", "volume")
        self.preclose_col = config.get("preclose_col", "preclose")
        self.limit_pct = config.get("limit_pct", 0.1)
        self.rebalance_days = config.get("rebalance_days", 1)  # 调仓天数，默认为1（每日调仓）

        self.portfolio_history = None

    def _run_backtest_internal(self, signals:Union[pl.DataFrame,pd.DataFrame]) -> Union[pl.DataFrame, None]:
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
            weight_col=self.weight_col,
            volume_col=self.volume_col,
            preclose_col=self.preclose_col,
            limit_pct=self.limit_pct,
            rebalance_days=self.rebalance_days
        )
        if isinstance(signals, pd.DataFrame):
            signals_pl = pl.from_pandas(signals)
        elif isinstance(signals, pl.DataFrame):
            signals_pl = signals
        else:
            print("错误: 信号数据必须是 Pandas DataFrame 或 Polars DataFrame。")
            return None

        try:
            result_pl = run_vectorized_backtest_rs(signals_pl, self.data_path, config)
            return result_pl
        except Exception as e:
            print(f"错误: 调用 Rust 回测函数时发生异常: {e}")
            return None

    def run_backtest(self):
        """
        执行回测。
        通过直接调用 Rust 函数来运行高性能的回测计算。
        """
        signals = self.strategy.generate_signals_for_all_dates()
        if signals is None:
            print("没有可用的交易信号数据。")
            return None
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

        if isinstance(portfolio_history, pd.DataFrame):
            df = portfolio_history
        elif isinstance(portfolio_history, pl.DataFrame):
            df = portfolio_history.to_pandas()

        else:
            print("错误：投资组合历史数据必须是 Pandas DataFrame 或 Polars DataFrame。")
            return None

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
        risk_free_rate = 0.01

        excess_return_annualized = annualized_return - risk_free_rate
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
        if isinstance(portfolio_history, pd.DataFrame):
            df = portfolio_history
        elif isinstance(portfolio_history, pl.DataFrame):
            df = portfolio_history.to_pandas()
        else:
            print("错误：投资组合历史数据必须是 Pandas DataFrame 或 Polars DataFrame。")
            return None
        
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
        if all_signals is None:
            print("没有可用的交易信号数据。")
            return None

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

def ensure_market_data_continuity(df: pl.DataFrame) -> pl.DataFrame:
    """
    修复行情数据：确保所有股票在所有交易日都有数据。
    缺失的行将用上一日的收盘价填充，Volume设为0。
    """
    print("正在进行行情数据完整性检查与填充...")
    
    # 1. 获取全量的日期和代码
    # 注意：这里假设 df 至少包含所有交易日。如果不全，需要外部传入 calendar list。
    unique_dates = df.select(pl.col("date").unique()).sort("date")
    unique_codes = df.select(pl.col("code").unique())
    
    # 2. 生成笛卡尔积（所有日期 x 所有代码）
    # cross join 会生成巨大的表，但这是保证连续性的唯一方法
    full_skeleton = unique_dates.join(unique_codes, how="cross")
    
    # 3. 将原始数据 Left Join 回骨架
    # 使用 join_asof 也可以，但在多代码场景下 left join + forward_fill 更直观
    merged = full_skeleton.join(df, on=["date", "code"], how="left")
    
    # 4. 执行前向填充 (Forward Fill)
    # 必须先按 code, date 排序
    df_filled = (
        merged
        .sort(["code", "date"])
        .with_columns([
            # 价格类字段：用前值填充 (模拟停牌时的价格)
            pl.col("close").forward_fill(),
            pl.col("preclose").forward_fill(),
            pl.col("open").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            
            # 成交量字段：缺失的填充为 0
            pl.col("volume").fill_null(0),
            pl.col("amount").fill_null(0.0),
        ])
        # 5. 清理上市前的数据
        # Forward fill 会把 IPO 之前的日期填成 null，这些行应该删掉
        .drop_nulls(subset=["close"])
    )
    
    # 6. 处理填充后仍然是 Null 的开盘价/最高价等（如果原始数据就有缺失）
    # 用 close 补齐 open/high/low
    df_final = df_filled.with_columns([
        pl.col("open").fill_null(pl.col("close")),
        pl.col("high").fill_null(pl.col("close")),
        pl.col("low").fill_null(pl.col("close")),
    ])
    
    print(f"数据填充完成。原始行数: {df.height}, 填充后行数: {df_final.height}")
    return df_final