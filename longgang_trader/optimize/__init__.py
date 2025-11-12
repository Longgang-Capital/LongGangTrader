"""
组合优化模块
提供各种组合优化算法，用于将因子信号转换为最优的投资组合权重
"""

# 旧版本接口（保持向后兼容）

# 新版本接口（与现有workflow兼容）
from .optimizer import (
    BasePortfolioOptimizer,
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    LayeredOptimizer
)

__all__ = [
    # 新版本接口
    'BasePortfolioOptimizer',
    'EqualWeightOptimizer',
    'MeanVarianceOptimizer',
    'RiskParityOptimizer',
    'LayeredOptimizer'
]