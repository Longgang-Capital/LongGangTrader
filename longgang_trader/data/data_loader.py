class DataLoader:
    """
    数据加载模块，负责从不同数据源（如CSV, 数据库）加载数据。
    """
    def __init__(self, config):
        """
        初始化数据加载器。
        :param config: 数据源配置，如文件路径、数据库连接信息等。
        """
        pass

    def load_stock_data(self, stock_codes, start_date, end_date, fields):
        """
        加载指定的股票行情数据。
        :param stock_codes: 股票代码列表。
        :param start_date: 开始日期。
        :param end_date: 结束日期。
        :param fields: 需要加载的字段，如 'open', 'high', 'low', 'close', 'volume'。
        :return: pandas.DataFrame，包含所有股票的数据。
        """
        pass

    def load_index_data(self, index_code, start_date, end_date):
        """
        加载指数行情数据。
        """
        pass

    def load_financial_data(self, stock_codes, report_date):
        """
        加载财务数据。
        """
        pass
