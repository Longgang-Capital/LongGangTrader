import pandas as pd
import polars as pl

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
    def __init__(self, strategy, data, config:dict):
        """
        :param strategy: 交易策略对象。
        :param data: 行情数据 (Pandas DataFrame)。
        :param initial_capital: 初始资金。
        :param transaction_cost: 交易成本（手续费、滑点等）。
        """
        self.strategy = strategy
        self.data = data
        self.config = config
        self.initial_capital = config.get("initial_capital", 1_000_000)
        self.transaction_cost = config.get("transaction_cost", 0.001)  # 0.1%
        self.symbol_col = config.get("symbol_col", "symbol")
        self.date_col = config.get("date_col", "date")
        self.close_col = config.get("close_col", "close")
        self.weight_col = config.get("weight_col", "target_weight")

        self.portfolio_history = None

    def run_backtest(self):
        """
        执行回测。
        通过直接调用 Rust 函数来运行高性能的回测计算。
        """
        if run_vectorized_backtest_rs is None: # type: ignore
            return

        # 1. 准备传递给 Rust 函数的数据
        signals = self.strategy.generate_signals_for_all_dates() # 假设策略返回 Pandas DataFrame

        # 创建回测配置对象
        config = BacktestConfig(
            initial_capital=self.initial_capital,
            transaction_cost_pct=self.transaction_cost,
            symbol_col=self.symbol_col,
            date_col=self.date_col,
            close_col=self.close_col,
            weight_col=self.weight_col
        ) # pyright: ignore[reportOptionalCall]
        # 将 Pandas DataFrame 转换为 Polars LazyFrame
        signals_pl = pl.from_pandas(signals).lazy()
        data_pl = pl.from_pandas(self.data).lazy()

        # 2. 直接调用 Rust 函数，传递 Polars LazyFrame
        try:
            # pyo3-polars 会处理 Polars LazyFrame 的高效传递
            result_pl = run_vectorized_backtest_rs(signals_pl, data_pl, config) # type: ignore
            # 将返回的 Polars DataFrame 转换回 Pandas DataFrame
            self.portfolio_history = result_pl.to_pandas()
            print("Rust 回测成功完成。")

        except Exception as e:
            print(f"错误: 调用 Rust 回测函数时发生异常: {e}")

    def get_portfolio_history(self):
        """
        获取投资组合的每日市值、持仓等历史记录。
        """
        return self.portfolio_history

    def calculate_performance_metrics(self):
        """
        计算回测的各项性能指标。
        :return: 包含年化收益、夏普比率、最大回撤等指标的字典。
        """
        if self.portfolio_history is None:
            print("请先运行回测。")
            return None
        # ... 此处添加性能计算逻辑 ...
        pass

    def plot_equity_curve(self):
        """
        绘制资金曲线。
        """
        if self.portfolio_history is None:
            print("请先运行回测。")
            return
        # ... 此处添加绘图逻辑 ...
        pass


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
