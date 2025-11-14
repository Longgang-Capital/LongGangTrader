class Preprocessor:
    """
    数据预处理模块，负责数据清洗、标准化等。
    """
    def __init__(self, data):
        """
        :param data: 原始数据 (pandas.DataFrame)。
        """
        pass

    def handle_missing_values(self, method='ffill'):
        """
        处理缺失值。
        :param method: 处理方法，如 'ffill' (向前填充), 'bfill' (向后填充), 'mean' (均值填充)。
        """
        pass

    def handle_outliers(self, method='winsorize'):
        """
        处理异常值。
        :param method: 处理方法，如 'winsorize' (缩尾处理)。
        """
        pass

    def normalize_data(self):
        """
        数据归一化。
        """
        pass

    def standardize_data(self):
        """
        数据标准化（去均值，除以标准差）。
        """
        pass

    def get_processed_data(self):
        """
        返回处理后的数据。
        """
        pass
