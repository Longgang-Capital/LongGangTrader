class BaseFactor:
    """
    所有因子的基类，定义了因子的基本接口。
    """
    def __init__(self, name):
        self.name = name

    def compute(self, data):
        """
        计算因子值。每个具体的因子类都需要实现这个方法。
        :param data: 包含所需字段的 pandas.DataFrame。
        :return: pandas.DataFrame，包含计算出的因子值。
        """
        raise NotImplementedError("每个因子都需要实现 compute 方法")


class MomentumFactor(BaseFactor):
    """
    动量因子示例。
    """
    def __init__(self, window):
        super().__init__(name=f"Momentum_{window}")
        self.window = window

    def compute(self, data):
        pass


class ValueFactor(BaseFactor):
    """
    价值因子示例，如市盈率倒数 (1/PE)。
    """
    def __init__(self):
        super().__init__(name="EP_Ratio")

    def compute(self, data):
        pass


class FactorCalculator:
    """
    用于批量计算和管理多个因子。
    """
    def __init__(self, data):
        self.data = data
        self.factors = []

    def add_factor(self, factor: BaseFactor):
        """
        添加一个要计算的因子。
        """
        pass

    def run(self):
        """
        批量计算所有已添加的因子。
        :return: pandas.DataFrame，包含所有计算出的因子值。
        """
        pass
