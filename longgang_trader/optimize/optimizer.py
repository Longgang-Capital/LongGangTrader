"""
优化后的组合优化模块
使用polars和向量化方法提升性能
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import polars as pl
import numpy as np


class BasePortfolioOptimizer(ABC):
    """
    组合优化器基类
    使用polars进行高性能数据处理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化器

        :param config: 优化器配置参数
        """
        self.config = config or {}
        self.optimization_info = {}

    @abstractmethod
    def optimize_portfolio(
        self,
        stock_data: pl.DataFrame,
        factor_data: pl.DataFrame,
        group_col: str = "group",
        date_col: str = "date",
        symbol_col: str = "order_book_id",
        factor_value_col: str = "factor_value",
        weight_col: str = "optimized_weight"
    ) -> pl.DataFrame:
        """
        执行组合优化

        :param stock_data: 市场数据DataFrame，包含价格、成交量等信息
        :param factor_data: 因子数据DataFrame，包含因子值和分组信息
        :param group_col: 分组列名
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :param factor_value_col: 因子值列名
        :param weight_col: 权重列名
        :return: 包含优化权重的DataFrame，格式与factor_data类似
        """
        pass

    def _prepare_data(
        self,
        stock_data: pl.DataFrame,
        factor_data: pl.DataFrame,
        date_col: str,
        symbol_col: str
    ) -> pl.DataFrame:
        """
        准备优化所需的数据

        :param stock_data: 市场数据
        :param factor_data: 因子数据
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :return: 合并后的数据
        """
        # 使用LazyFrame进行高效合并
        factor_lazy = factor_data.lazy()
        stock_lazy = stock_data.select([date_col, symbol_col, "close"]).lazy()

        # 合并数据
        merged_data = factor_lazy.join(
            stock_lazy,
            on=[date_col, symbol_col],
            how="left"
        ).collect()

        return merged_data

    def _calculate_returns(self, data: pl.DataFrame, symbol_col: str, date_col: str) -> pl.DataFrame:
        """
        计算收益率 - 使用polars向量化方法

        :param data: 合并后的数据
        :param symbol_col: 股票代码列名
        :param date_col: 日期列名
        :return: 收益率DataFrame
        """
        # 按股票代码分组，计算收益率
        returns_data = data.lazy().sort([symbol_col, date_col]).with_columns([
            pl.col("close").pct_change().over(symbol_col).alias("return")
        ]).collect()

        return returns_data

    def _normalize_weights(self, data: pl.DataFrame, date_col: str, weight_col: str) -> pl.DataFrame:
        """
        标准化权重，确保每个日期的权重和为1

        :param data: 权重数据
        :param date_col: 日期列名
        :param weight_col: 权重列名
        :return: 标准化后的数据
        """
        # 使用polars的窗口函数进行向量化标准化
        normalized_data = data.lazy().with_columns([
            (pl.col(weight_col) / pl.col(weight_col).sum().over(date_col)).alias(weight_col)
        ]).collect()

        return normalized_data


class EqualWeightOptimizer(BasePortfolioOptimizer):
    """
    等权重优化器
    在选定的股票池中分配相等的权重
    """

    def optimize_portfolio(
        self,
        stock_data: pl.DataFrame,
        factor_data: pl.DataFrame,
        group_col: str = "group",
        date_col: str = "date",
        symbol_col: str = "order_book_id",
        factor_value_col: str = "factor_value",
        weight_col: str = "optimized_weight"
    ) -> pl.DataFrame:
        """
        执行等权重优化 - 使用polars向量化方法

        :param stock_data: 市场数据
        :param factor_data: 因子数据
        :param group_col: 分组列名
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :param factor_value_col: 因子值列名
        :param weight_col: 权重列名
        :return: 包含等权重的DataFrame
        """
        result_data = factor_data.clone()

        # 使用polars的窗口函数进行向量化等权重分配
        if group_col in factor_data.columns:
            # 按日期和分组计算等权重
            result_data = result_data.lazy().with_columns([
                (1.0 / pl.col(symbol_col).count().over([date_col, group_col])).alias(weight_col)
            ]).collect()
        else:
            # 按日期计算等权重
            result_data = result_data.lazy().with_columns([
                (1.0 / pl.col(symbol_col).count().over(date_col)).alias(weight_col)
            ]).collect()

        self.optimization_info = {
            "optimizer_type": "EqualWeight",
            "total_dates": result_data[date_col].n_unique(),
            "average_stocks_per_date": len(result_data) / result_data[date_col].n_unique()
        }

        return result_data


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """
    均值-方差优化器
    基于Markowitz现代投资组合理论，使用解析解
    """

    def __init__(self, config: Dict[str, Any] = {}):
        super().__init__(config)
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        self.max_weight = config.get("max_weight", 0.1)  # 单只股票最大权重
        self.min_weight = config.get("min_weight", 0.0)  # 单只股票最小权重
        self.risk_aversion = config.get("risk_aversion", 1.0)  # 风险厌恶系数

    def optimize_portfolio(
        self,
        stock_data: pl.DataFrame,
        factor_data: pl.DataFrame,
        group_col: str = "group",
        date_col: str = "date",
        symbol_col: str = "order_book_id",
        factor_value_col: str = "factor_value",
        weight_col: str = "optimized_weight"
    ) -> pl.DataFrame:
        """
        执行均值-方差优化 - 使用解析解和向量化方法

        :param stock_data: 市场数据
        :param factor_data: 因子数据
        :param group_col: 分组列名
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :param factor_value_col: 因子值列名
        :param weight_col: 权重列名
        :return: 包含优化权重的DataFrame
        """
        # 准备数据
        merged_data = self._prepare_data(stock_data, factor_data, date_col, symbol_col)
        returns_data = self._calculate_returns(merged_data, symbol_col, date_col)

        result_data = factor_data.clone()

        # 获取所有日期
        dates = result_data[date_col].unique().sort()

        # 预计算所有股票的历史收益率数据
        historical_returns_dict = self._precompute_historical_returns(
            returns_data, dates, lookback_period=60
        )

        # 为所有日期批量计算优化权重
        optimized_weights_list = []

        for date in dates:
            date_symbols = result_data.filter(pl.col(date_col) == date)[symbol_col].to_list()

            if len(date_symbols) == 0:
                continue

            # 获取历史收益率数据
            historical_returns = historical_returns_dict.get(date)

            if historical_returns is None or len(historical_returns.columns) == 0:
                # 如果没有足够的历史数据，使用等权重
                n_stocks = len(date_symbols)
                equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0.0
                date_weights = {symbol: equal_weight for symbol in date_symbols}
            else:
                # 使用解析解进行均值-方差优化
                try:
                    date_weights = self._mean_variance_analytical_solution(
                        historical_returns, date_symbols
                    )
                except Exception as e:
                    print(f"日期 {date} 优化失败: {e}")
                    # 优化失败时使用等权重
                    n_stocks = len(date_symbols)
                    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0.0
                    date_weights = {symbol: equal_weight for symbol in date_symbols}

            # 添加日期和权重信息
            for symbol, weight in date_weights.items():
                optimized_weights_list.append({
                    date_col: date,
                    symbol_col: symbol,
                    weight_col: weight
                })

        # 创建优化权重DataFrame
        optimized_weights_df = pl.DataFrame(optimized_weights_list)

        # 确保日期数据类型一致
        result_data = result_data.with_columns([
            pl.col(date_col).cast(pl.Datetime)
        ])
        optimized_weights_df = optimized_weights_df.with_columns([
            pl.col(date_col).cast(pl.Datetime)
        ])

        # 合并回原始数据
        result_data = result_data.join(
            optimized_weights_df,
            on=[date_col, symbol_col],
            how="left"
        ).fill_null(0.0)

        # 确保权重和为1
        result_data = self._normalize_weights(result_data, date_col, weight_col)

        self.optimization_info = {
            "optimizer_type": "MeanVariance",
            "risk_free_rate": self.risk_free_rate,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "risk_aversion": self.risk_aversion
        }

        return result_data

    def _precompute_historical_returns(
        self,
        returns_data: pl.DataFrame,
        dates: pl.Series,
        lookback_period: int = 60
    ) -> Dict[Any, pl.DataFrame]:
        """
        预计算所有日期的历史收益率数据

        :param returns_data: 收益率数据
        :param dates: 日期序列
        :param lookback_period: 回溯期
        :return: 历史收益率数据字典
        """
        import pandas as pd

        historical_returns_dict = {}

        # 转换为pandas进行高效的时间序列操作
        returns_pd = returns_data.to_pandas()
        # 确保日期列正确解析
        date_col_name = returns_data.columns[0]
        if date_col_name == "date":
            returns_pd[date_col_name] = pd.to_datetime(returns_pd[date_col_name])

        for date in dates:
            date_pd = pd.to_datetime(date)
            start_date = date_pd - pd.Timedelta(days=lookback_period)

            # 筛选历史数据
            mask = (returns_pd["date"] >= start_date) & (returns_pd["date"] < date_pd)
            historical_data = returns_pd[mask]

            if historical_data.empty:
                historical_returns_dict[date] = None
                continue

            # 转换为宽格式
            returns_wide = historical_data.pivot_table(
                index="date",
                columns="order_book_id",
                values="return"
            ).dropna(axis=1, how="all")

            # 转换回polars
            if not returns_wide.empty:
                historical_returns_dict[date] = pl.from_pandas(returns_wide.reset_index())
            else:
                historical_returns_dict[date] = None

        return historical_returns_dict

    def _mean_variance_analytical_solution(
        self,
        historical_returns: pl.DataFrame,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        均值-方差优化的解析解
        使用公式: w = λ * Σ^(-1) * μ

        :param historical_returns: 历史收益率数据
        :param symbols: 股票代码列表
        :return: 优化权重字典
        """
        # 转换为numpy数组
        returns_array = historical_returns.select(symbols).to_numpy()

        # 计算预期收益率和协方差矩阵
        expected_returns = np.nanmean(returns_array, axis=0)
        covariance_matrix = np.cov(returns_array, rowvar=False)

        # 处理协方差矩阵的奇异性
        try:
            # 添加小的正则化项避免奇异性
            regularization = 1e-6 * np.eye(covariance_matrix.shape[0])
            cov_inv = np.linalg.inv(covariance_matrix + regularization)
        except np.linalg.LinAlgError:
            # 如果仍然奇异，使用等权重
            n_assets = len(symbols)
            equal_weight = 1.0 / n_assets
            return {symbol: equal_weight for symbol in symbols}

        # 使用解析解计算权重
        # w = λ * Σ^(-1) * μ
        weights = self.risk_aversion * cov_inv @ expected_returns

        # 应用权重约束
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # 标准化权重
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # 如果所有权重为0，使用等权重
            weights = np.ones(len(symbols)) / len(symbols)

        # 创建权重字典
        weights_dict = {symbol: weight for symbol, weight in zip(symbols, weights)}

        return weights_dict


class RiskParityOptimizer(BasePortfolioOptimizer):
    """
    风险平价优化器
    根据资产对组合风险的贡献度分配权重
    """

    def optimize_portfolio(
        self,
        stock_data: pl.DataFrame,
        factor_data: pl.DataFrame,
        group_col: str = "group",
        date_col: str = "date",
        symbol_col: str = "order_book_id",
        factor_value_col: str = "factor_value",
        weight_col: str = "optimized_weight"
    ) -> pl.DataFrame:
        """
        执行风险平价优化

        :param stock_data: 市场数据
        :param factor_data: 因子数据
        :param group_col: 分组列名
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :param factor_value_col: 因子值列名
        :param weight_col: 权重列名
        :return: 包含优化权重的DataFrame
        """
        # 简化实现：使用等权重
        equal_weight_optimizer = EqualWeightOptimizer()
        return equal_weight_optimizer.optimize_portfolio(
            stock_data, factor_data, group_col, date_col, symbol_col, factor_value_col, weight_col
        )


class LayeredOptimizer:
    """
    分层优化器
    结合因子分组和组合优化的完整流程
    """

    def __init__(self, optimizer: BasePortfolioOptimizer, config: Dict[str, Any] = {}):
        """
        初始化分层优化器

        :param optimizer: 底层优化器
        :param config: 配置参数
        """
        self.optimizer = optimizer
        self.config = config or {}
        self.n_groups = config.get("n_groups", 5)  # 分层数量
        self.group_col = config.get("group_col", "group")

    def create_factor_groups(
        self,
        factor_data: pl.DataFrame,
        date_col: str = "date",
        symbol_col: str = "order_book_id",
        factor_value_col: str = "factor_value"
    ) -> pl.DataFrame:
        """
        创建因子分组 - 使用polars向量化方法

        :param factor_data: 因子数据
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :param factor_value_col: 因子值列名
        :return: 包含分组信息的因子数据
        """
        result_data = factor_data.clone()

        # 使用rank排序进行无重复分组
        result_data = result_data.lazy().with_columns([
            # 按因子值降序排列（因子值越大越好）
            pl.col(factor_value_col)
            .rank(method="dense", descending=True)
            .over(date_col)
            .alias("rank")
        ]).with_columns([
            # 根据排名分配分组
            ((pl.col("rank") - 1) * self.n_groups / pl.col("rank").max().over(date_col))
            .floor()
            .cast(pl.Int64)
            .alias(self.group_col)
        ]).drop("rank").collect()

        return result_data

    def optimize_layered_portfolio(
        self,
        stock_data: pl.DataFrame,
        factor_data: pl.DataFrame,
        date_col: str = "date",
        symbol_col: str = "order_book_id",
        factor_value_col: str = "factor_value",
        weight_col: str = "optimized_weight"
    ) -> pl.DataFrame:
        """
        执行分层优化

        :param stock_data: 市场数据
        :param factor_data: 因子数据
        :param date_col: 日期列名
        :param symbol_col: 股票代码列名
        :param factor_value_col: 因子值列名
        :param weight_col: 权重列名
        :return: 包含优化权重的DataFrame
        """
        # 1. 创建因子分组
        grouped_factor_data = self.create_factor_groups(
            factor_data, date_col, symbol_col, factor_value_col
        )

        # 2. 在每组内执行组合优化
        optimized_data = self.optimizer.optimize_portfolio(
            stock_data,
            grouped_factor_data,
            self.group_col,
            date_col,
            symbol_col,
            factor_value_col,
            weight_col
        )

        return optimized_data