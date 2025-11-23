# file: main_trader.py

import sys
from time import sleep
from typing import Any

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import (
    TickData, OrderData, TradeData, ContractData,
    SubscribeRequest, OrderRequest, CancelRequest
)
from vnpy.trader.event import (
    EVENT_TICK, EVENT_ORDER, EVENT_TRADE, EVENT_CONTRACT
)
from vnpy.trader.constant import Direction, OrderType, Exchange

# 引入 vn.py 的 RQData 接口
#from vnpy.gateway.rqdata import RqdataGateway
from vnpy_rqdata.rqdata_gateway import RqdataGateway
# 引入 vn.py 的 CTA 策略应用（为了方便管理和查看策略状态，可选）
#from vnpy.app.cta_strategy import CtaStrategyApp

# ===================================================================
# 假设这是您已经开发好的、独立的多因子框架核心
# ===================================================================
class MyFactorFramework:
    """
    一个示例性的、您自己的多因子框架。
    它独立于vn.py，只负责计算因子和产生交易信号。
    """
    def __init__(self, main_engine: MainEngine):
        print("我的多因子框架已初始化。")
        self.main_engine = main_engine
        self.subscribed_symbols = set() # 跟踪已订阅的合约
        self.last_tick = None

    def on_realtime_tick(self, tick: TickData):
        """
        这是框架处理实时行情的核心入口。
        """
        print(f"【我的框架】接收到行情：{tick.vt_symbol}, 最新价: {tick.last_price}")
        self.last_tick = tick

        # -------- 在这里执行您的多因子逻辑 --------
        # 1. 更新因子值
        # 2. 计算模型得分
        # 3. 判断是否满足交易条件
        # 示例：一个非常简单的逻辑，当价格大于某个值时买入
        if tick.last_price > 50 and tick.vt_symbol not in self.get_all_positions():
             print(f"【我的框架】产生买入信号！最新价 {tick.last_price} > 50")
             self.execute_buy_order(tick.vt_symbol, tick.last_price, 100)

    def on_order_update(self, order: OrderData):
        """处理委托回报"""
        print(f"【我的框架】收到委托更新: {order.vt_symbol}, 状态: {order.status.value}, 信息: {order.msg}")

    def on_trade_update(self, trade: TradeData):
        """处理成交通知"""
        print(f"【我的框架】收到成交通知: {trade.vt_symbol}, 方向: {trade.direction.value}, 成交价: {trade.price}, 数量: {trade.volume}")

    def subscribe_market_data(self, symbol: str, exchange: Exchange):
        """
        订阅行情。这是一个封装好的动作，内部调用vn.py的接口。
        """
        if symbol in self.subscribed_symbols:
            return
        print(f"【我的框架】准备订阅合约: {symbol}.{exchange.value}")
        req = SubscribeRequest(symbol=symbol, exchange=exchange)
        self.main_engine.subscribe(req, "RQDATA")
        self.subscribed_symbols.add(symbol)
        
    def execute_buy_order(self, vt_symbol: str, price: float, volume: int):
        """
        执行买入委托。这也是一个封装好的动作。
        """
        # 从 vt_symbol 解析出 symbol 和 exchange
        symbol, exchange_str = vt_symbol.split(".")
        exchange = Exchange(exchange_str)

        req = OrderRequest(
            symbol=symbol,
            exchange=exchange,
            direction=Direction.LONG,
            type=OrderType.LIMIT, # 使用限价单
            volume=volume,
            price=price,
            reference="MyFactorFramework_Buy" # 自定义引用
        )
        orderid = self.main_engine.send_order(req, "RQDATA")
        print(f"【我的框架】已发送买入委托，ID: {orderid}")

    def get_all_positions(self):
        """获取所有持仓（仅为示例，实际应更完善）"""
        # 实际应用中可以从 main_engine 获取持仓信息
        return []

# ===================================================================
# 主程序：负责启动和粘合 vn.py 与您的框架
# ===================================================================
def main():
    """主函数"""
    # 1. 初始化 vn.py 核心引擎
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # 2. 添加 RQData 接口和 CTA 策略应用
    main_engine.add_gateway(RqdataGateway)
    #main_engine.add_app(CtaStrategyApp) # 添加App不是必须的，但通常建议这样做

    # 3. 配置并连接 RQData
    rqdata_setting = {
        "用户名": "your_rqdata_phone_number", # 必须是手机号
        "密码": "your_rqdata_service_password" # 必须是数据服务密码
    }
    main_engine.connect(rqdata_setting, "RQDATA")
    print("等待5秒确保RQData连接和初始化...")
    sleep(5)

    # 4. 初始化您自己的多因子框架
    my_framework = MyFactorFramework(main_engine)

    # 5. 定义事件处理函数（“胶水”代码）
    def process_tick_event(event: Event):
        tick_data: TickData = event.data
        my_framework.on_realtime_tick(tick_data)

    def process_order_event(event: Event):
        order_data: OrderData = event.data
        my_framework.on_order_update(order_data)

    def process_trade_event(event: Event):
        trade_data: TradeData = event.data
        my_framework.on_trade_update(trade_data)

    # 6. 将事件处理函数注册到 vn.py 的事件引擎
    event_engine.register(EVENT_TICK, process_tick_event)
    event_engine.register(EVENT_ORDER, process_order_event)
    event_engine.register(EVENT_TRADE, process_trade_event)
    print("事件处理函数注册完成。")

    # 7. 让您的框架发起订阅
    my_framework.subscribe_market_data(symbol="000001", exchange=Exchange.SZSE)
    my_framework.subscribe_market_data(symbol="600519", exchange=Exchange.SSE)

    # 8. 启动主循环，程序开始接收事件
    print("系统启动，开始接收实时行情... 按 Ctrl+C 退出")
    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            print("程序退出")
            main_engine.close()
            sys.exit(0)

if __name__ == "__main__":
    main()