import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Optional, Union



class BaseStrategy:
    """
    交易策略基类。
    """

    def __init__(
        self,
        factor_data: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}

        if isinstance(factor_data, pd.DataFrame):
            factor_data = pl.from_pandas(factor_data)
        elif isinstance(factor_data, pl.LazyFrame):
            factor_data = factor_data.collect()
        elif not isinstance(factor_data, pl.DataFrame):
            raise TypeError("factor_data must be a pandas or polars DataFrame")

        self.factor_data = factor_data.clone()
        self.config = config
        self.date_col = config.get("date_col", "date")
        self.symbol_col = config.get("symbol_col", "order_book_id")
        self.group_col = config.get("group_col", "group")

        factor_key = config.get("factor_value_col", None)
        weight_key = config.get("weight_col", None)
        if factor_key and not weight_key:
            weight_key = factor_key
        elif weight_key and not factor_key:
            factor_key = weight_key
        self.factor_value_col = factor_key or "factor_value"
        self.weight_col = weight_key or "target_weight"

    def generate_signals(self, current_date, current_positions):
        """
        根据因子值生成交易信号（目标持仓）。
        :param current_date: 当前交易日。
        :param current_positions: 当前持仓。
        :return: 目标持仓权重 (dict)。
        """
        raise NotImplementedError

    def generate_signals_for_all_dates(self)-> Union[pl.DataFrame, None]:
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
        data = self.factor_data.clone()

        # 在每个日期，找到因子值最高的股票
        signals = (
            data
            .sort([self.date_col, self.factor_value_col], descending=[False, True])
            .group_by(self.date_col)
            .head(1)
            .with_columns(pl.lit(1.0).alias(self.weight_col))
            .select([self.date_col, self.symbol_col, self.weight_col])
        )

        print("\n生成的交易信号 (signals):")
        print(signals)
        return signals


class TopNFactorStrategy(BaseStrategy):
    """
    投资因子值前N的交易策略。
    在每个交易日，选择因子值最高的N只股票进行等权重投资。
    """
    def __init__(self, factor_data, config: Optional[dict] = None):
        super().__init__(factor_data, config)
        # 获取配置参数，默认为前5只股票
        self.top_n = self.config.get("top_n", 5)

    def generate_signals_for_all_dates(self):
        """
        为所有日期一次性生成交易信号。
        在每个交易日，选择因子值最高的N只股票，进行等权重分配。
        """
        # 复制因子数据以避免修改原始数据
        data = self.factor_data.clone()

        top_stocks = (
            data
            .sort([self.date_col, self.factor_value_col], descending=[False, True])
            .group_by(self.date_col)
            .head(self.top_n)
            .with_columns(pl.lit(1.0 / self.top_n).alias(self.weight_col))
            .select([self.date_col, self.symbol_col, self.weight_col])
        )

        total_days = (
            top_stocks.select(pl.col(self.date_col).n_unique()).item()
            if not top_stocks.is_empty()
            else 0
        )
        total_signals = top_stocks.height
        avg_per_day = total_signals / total_days if total_days else 0.0

        print(f"\n生成的交易信号 (Top {self.top_n} Factor Strategy):")
        print(f"总交易日数: {total_days}")
        print(f"总信号数: {total_signals}")
        print(f"平均每个交易日选择的股票数: {avg_per_day:.2f}")
        print("信号数据预览:")
        print(top_stocks.head())

        return top_stocks

    def generate_signals(self, current_date, current_positions):
        """
        根据当前日期和持仓生成交易信号。
        :param current_date: 当前交易日
        :param current_positions: 当前持仓
        :return: 目标持仓权重 (dict)
        """
        # 获取当前日期的因子数据
        current_data = self.factor_data.filter(pl.col(self.date_col) == current_date)

        if current_data.is_empty():
            return {}

        # 按因子值降序排列，取前N只股票
        top_stocks = current_data.sort(self.factor_value_col, descending=True).head(self.top_n)

        # 计算等权重
        target_weights = {}
        weight_per_stock = 1.0 / top_stocks.height if top_stocks.height > 0 else 0.0

        for row in top_stocks.iter_rows(named=True):
            symbol = row[self.symbol_col]
            target_weights[symbol] = weight_per_stock

        return target_weights


class GroupedFactorStrategy(BaseStrategy):
    """
    一个简单的策略，在每个交易日，使用优化权重买入各分组内的股票。
    该策略假设优化器已经计算好了每个股票的目标权重。
    """
    def __init__(self, factor_data, config: Optional[dict] = None):
        super().__init__(factor_data, config)

        self.optimized_weight_col = self.config.get("weight_col", "optimized_weight")


    def generate_signals_for_all_dates(self):
        # The optimizer has already calculated weights, so this strategy just formats the data
        signals = self.factor_data.clone()
        if self.optimized_weight_col not in signals.columns:
            raise ValueError(
                f"列 '{self.optimized_weight_col}' 不存在于输入数据中，无法生成信号。"
            )
        signals = signals.rename({self.optimized_weight_col: self.weight_col})
        return signals


class SimpleWeightStrategy(BaseStrategy):
    """
    简单权重策略，直接使用原始权重数据，不进行分层也不修改权重。
    假设输入数据已经包含了权重列。
    """
    def __init__(self, factor_data, config: Optional[dict] = None):
        super().__init__(factor_data, config)
        # 权重列名，默认为"weight"
        self.weight_col = self.config.get("weight_col", "weight")

    def generate_signals_for_all_dates(self):
        """
        为所有日期生成交易信号。
        直接使用原始权重数据，不进行任何修改。
        """
        # 复制因子数据以避免修改原始数据
        signals = self.factor_data.clone()

        # 检查权重列是否存在
        if self.weight_col not in signals.columns:
            raise ValueError(f"权重列 '{self.weight_col}' 不存在于输入数据中。")

        # 选择需要的列
        signals = signals.select([self.date_col, self.symbol_col, self.weight_col])

        total_days = (
            signals.select(pl.col(self.date_col).n_unique()).item()
            if not signals.is_empty()
            else 0
        )
        total_signals = signals.height

        print(f"\n生成的交易信号 (Simple Weight Strategy):")
        print(f"总交易日数: {total_days}")
        print(f"总信号数: {total_signals}")
        print(f"权重列: {self.weight_col}")
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
        current_data = self.factor_data.filter(pl.col(self.date_col) == current_date)

        if current_data.is_empty():
            return {}

        # 检查权重列是否存在
        if self.weight_col not in current_data.columns:
            return {}

        # 创建权重字典
        target_weights = {
            row[self.symbol_col]: row[self.weight_col]
            for row in current_data.iter_rows(named=True)
        }

        return target_weights
    

class RollingTopQuantileStrategy(BaseStrategy):
    """
    基于滚动预测值分位数的信号加权策略。
    
    逻辑复刻：
    1. 对因子值进行 N 日滚动平均 (Rolling Mean)。
    2. 计算每日的截面 Q 分位数 (如 Top 10%)。
    3. 筛选出：(滚动值 >= 当日分位数) 且 (滚动值 > 0) 的股票。
    4. 按照预测值占比进行归一化加权 (Signal Strength Weighting)。
    """
    def __init__(self, factor_data, config: Optional[dict] = None):
        super().__init__(factor_data, config)
        # 获取配置参数
        self.rolling_window = self.config.get("rolling_window", 20)  # 默认20日平滑
        self.quantile = self.config.get("quantile", 0.9)             # 默认90%分位数
        self.min_score = self.config.get("min_score", 0.0)           # 绝对值门槛，原文逻辑为 > 0

    def generate_signals_for_all_dates(self):
        """
        为所有日期一次性生成信号 (推荐使用此方法，效率最高)。
        """
        # 2. 滚动平滑 (Rolling Mean)
        data = (
            self.factor_data
            .clone()
            .sort([self.symbol_col, self.date_col])
            .with_columns(
                pl.col(self.factor_value_col)
                .rolling_mean(
                    window_size=self.rolling_window,
                )
                .over(self.symbol_col)
                .alias("rolling_score")
            )
        )
        # 3. 计算每日的分位阈值
        thresholds = (
            data.group_by(self.date_col)
            .agg(
                pl.col("rolling_score")
                .drop_nulls()
                .quantile(self.quantile)
                .alias("_threshold")
            )
        )
        # 4. 构造筛选 Mask 并计算权重
        signals = (
            data.join(thresholds, on=self.date_col, how="left")
            .filter(
                pl.col("rolling_score").is_not_null()
                & (pl.col("rolling_score") >= pl.col("_threshold"))
                & (pl.col("rolling_score") > self.min_score)
            )
            .with_columns(
                pl.col("rolling_score").sum().over(self.date_col).alias("_score_sum")
            )
            .with_columns(
                pl.when(pl.col("_score_sum") > 0)# 避免除以0，如果某天没有股票入选，sum为0，div后会产生inf或nan
                .then(pl.col("rolling_score") / pl.col("_score_sum"))
                .otherwise(None)
                .alias(self.weight_col)# 计算归一化权重
            )
            .filter(pl.col(self.weight_col).is_not_null())
            .select([self.date_col, self.symbol_col, self.weight_col])
        )

        # 打印统计信息
        total_days = (
            signals.select(pl.col(self.date_col).n_unique()).item()
            if not signals.is_empty()
            else 0
        )
        avg_positions = signals.height / total_days if total_days else 0.0

        print(
            f"\n生成的交易信号 (Rolling Top {int((1 - self.quantile) * 100)}% Strategy):"
        )
        print(f"总交易日数: {total_days}")
        print(f"平均每日持仓数: {avg_positions:.2f}")
        print("信号数据预览:")
        print(signals.head())

        return signals

    def generate_signals(self, current_date, current_positions):
        """
        根据当前日期生成信号（单步模式）。
        注意：由于涉及滚动计算，单步模式需要回溯历史数据，效率较低。
        """
        # 找出截止到 current_date 的历史数据
        # 只需要取最近的 rolling_window * 2 天的数据足以计算（为了保险）
        # 实际工程中通常会预先计算好所有信号，这里为了演示逻辑：
        
        # 1. 筛选相关历史数据
        history_data = self.factor_data.filter(pl.col(self.date_col) <= current_date)

        # 性能优化：如果历史数据太长，只切片最近的一段
        unique_days = (
            history_data.select(pl.col(self.date_col).n_unique()).item()
            if not history_data.is_empty()
            else 0
        )
        if unique_days < self.rolling_window:
            return {}  # 数据不足以计算滚动均值

        history_with_scores = (
            history_data
            .sort([self.symbol_col, self.date_col])
            .with_columns(
                pl.col(self.factor_value_col)
                .rolling_mean(window_size=self.rolling_window)
                .over(self.symbol_col)
                .alias("rolling_score")
            )
        )

        current_slice = (
            history_with_scores
            .filter(pl.col(self.date_col) == current_date)
            .drop_nulls("rolling_score")
        )

        if current_slice.is_empty():
            return {}

        threshold_df = current_slice.select(
            pl.col("rolling_score").quantile(self.quantile)
        )
        threshold = threshold_df.item() if threshold_df.height else None

        if threshold is None:
            return {}

        selected = current_slice.filter(
            (pl.col("rolling_score") >= threshold)
            & (pl.col("rolling_score") > self.min_score)
        )

        if selected.is_empty():
            return {}

        total_score_df = selected.select(pl.col("rolling_score").sum())
        total_score = total_score_df.item() if total_score_df.height else None

        if total_score is None or total_score <= 0:
            return {}

        target_weights = {
            row[self.symbol_col]: row["rolling_score"] / total_score
            for row in selected.iter_rows(named=True)
        }

        return target_weights