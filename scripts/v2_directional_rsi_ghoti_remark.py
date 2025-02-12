import os
from decimal import Decimal
from typing import Dict, List, Optional

import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction

# 策略配置
class SimpleDirectionalRSIConfig(StrategyV2ConfigBase):
    # 获取当前脚本名
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    # 市场配置
    markets: Dict[str, List[str]] = {}
    # K 线配置
    candles_config: List[CandlesConfig] = []
    # 控制器配置
    controllers_config: List[str] = []
    # 交易所 默认hyperliquid_perpetual
    exchange: str = Field(default="hyperliquid_perpetual", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"))
    # 交易对 默认ETH-USD
    trading_pair: str = Field(default="ETH-USD", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair where the bot will trade"))
    # K线数据来源 默认binance_perpetual
    candles_exchange: str = Field(default="binance_perpetual", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Candles exchange used to calculate RSI"))
    # 交易对 默认ETH-USDT
    candles_pair: str = Field(default="ETH-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Candles trading pair used to calculate RSI"))
    # K线时间间隔 默认1m
    candles_interval: str = Field(default="1m", client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Candle interval (e.g. 1m for 1 minute)"))
    # K线数量 默认60
    candles_length: int = Field(default=60, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Number of candles used to calculate RSI (e.g. 60)"))
    # RSI下界 默认30 RSI 低于 30 进入多单
    rsi_low: float = Field(default=30, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "RSI lower bound to enter long position (e.g. 30)"))
    # RSI上界 默认70 RSI 高于 70 进入空单
    rsi_high: float = Field(default=70, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "RSI upper bound to enter short position (e.g. 70)"))
    # 交易量 默认30
    order_amount_quote: Decimal = Field(default=30, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount in quote asset"))
    # 杠杆 默认10
    leverage: int = Field(default=10, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Leverage (e.g. 10 for 10x)"))
    # 交易模式 默认ONEWAY(单向)
    position_mode: PositionMode = Field(default="ONEWAY", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Position mode (HEDGE/ONEWAY)"))

    # Triple Barrier Configuration 三重屏障
    # 止损 默认3%
    stop_loss: Decimal = Field(default=Decimal("0.03"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Position stop loss (e.g. 0.03 for 3%)"))
    # 止盈 默认1%
    take_profit: Decimal = Field(default=Decimal("0.01"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Position take profit (e.g. 0.01 for 1%)"))
    # 持仓上限 默认45分钟
    time_limit: int = Field(default=60 * 45, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Position time limit in seconds (e.g. 300 for 5 minutes)"))

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
            time_limit_order_type=OrderType.MARKET  # Defaulting to MARKET as per requirement
        )

    @validator('position_mode', pre=True, allow_reuse=True)
    def validate_position_mode(cls, v: str) -> PositionMode:
        if v.upper() in PositionMode.__members__:
            return PositionMode[v.upper()]
        raise ValueError(f"Invalid position mode: {v}. Valid options are: {', '.join(PositionMode.__members__)}")

# 策略实现
class SimpleDirectionalRSI(StrategyV2Base):
    """
    This strategy uses RSI (Relative Strength Index) to generate trading signals and execute trades based on the RSI values.
    It defines the specific parameters and configurations for the RSI strategy.
    """

    account_config_set = False

    # 初始化市场
    @classmethod
    def init_markets(cls, config: SimpleDirectionalRSIConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    # 设置交易市场
    def __init__(self, connectors: Dict[str, ConnectorBase], config: SimpleDirectionalRSIConfig):
        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_exchange,
                trading_pair=config.candles_pair,
                interval=config.candles_interval,
                max_records=config.candles_length + 10
            ))
        super().__init__(connectors, config)
        self.config = config
        self.current_rsi = None
        self.current_signal = None

    # 策略启动
    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    # 交易逻辑
    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        # 创建一个空列表用于存储交易执行动作
        create_actions = []
        
        # 获取交易信号（从配置的K线数据源获取RSI信号）
        # 信号值: 1表示做多信号，-1表示做空信号，None表示无信号
        signal = self.get_signal(self.config.candles_exchange, self.config.candles_pair)
        
        # 获取当前活跃的多空仓位
        # active_longs: 当前活跃的多仓列表
        # active_shorts: 当前活跃的空仓列表
        active_longs, active_shorts = self.get_active_executors_by_side(self.config.exchange,
                                                                        self.config.trading_pair)
         # 如果有交易信号
        if signal is not None:
            # 获取当前交易对的中间价格
            mid_price = self.market_data_provider.get_price_by_type(self.config.exchange,
                                                                    self.config.trading_pair,
                                                                    PriceType.MidPrice)
            # 如果是做多信号(1)且当前没有活跃的多仓
            if signal == 1 and len(active_longs) == 0:
                # 创建一个做多仓位执行器
                create_actions.append(CreateExecutorAction(
                    executor_config=PositionExecutorConfig(
                        timestamp=self.current_timestamp, # 当前时间戳
                        connector_name=self.config.exchange, # 交易所
                        trading_pair=self.config.trading_pair, # 交易对
                        side=TradeType.BUY, # 买入方向
                        entry_price=mid_price, # 入场价格为中间价
                        amount=self.config.order_amount_quote / mid_price, # 交易数量 = 交易量 / 中间价
                        triple_barrier_config=self.config.triple_barrier_config, # 三重屏障配置（止损/止盈/时间限制）
                        leverage=self.config.leverage # 杠杆倍数
                    )))
            # 如果是做空信号(-1)且当前没有活跃的空仓
            elif signal == -1 and len(active_shorts) == 0:
                # 创建一个做空仓位执行器
                create_actions.append(CreateExecutorAction(
                    executor_config=PositionExecutorConfig(
                        timestamp=self.current_timestamp, # 当前时间戳
                        connector_name=self.config.exchange, # 交易所
                        trading_pair=self.config.trading_pair, # 交易对
                        side=TradeType.SELL, # 卖出方向
                        entry_price=mid_price, # 入场价格为中间价
                        amount=self.config.order_amount_quote / mid_price, # 交易数量 = 交易量 / 中间价
                        triple_barrier_config=self.config.triple_barrier_config, # 三重屏障配置（止损/止盈/时间限制）
                        leverage=self.config.leverage # 杠杆倍数
                    )))
        return create_actions

    # 停止逻辑
    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        # 创建一个空列表用于存储停止执行动作
        stop_actions = []
        
        # 获取交易信号（从配置的K线数据源获取RSI信号）
        # 信号值: 1表示做多信号，-1表示做空信号，None表示无信号
        signal = self.get_signal(self.config.candles_exchange, self.config.candles_pair)
        active_longs, active_shorts = self.get_active_executors_by_side(self.config.exchange,
                                                                        self.config.trading_pair)
        # 如果信号有效
        if signal is not None:
            # 如果是做空信号(-1)且当前有活跃的多仓
            if signal == -1 and len(active_longs) > 0:
                stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_longs])
            # 如果是做多信号(1)且当前有活跃的空仓
            elif signal == 1 and len(active_shorts) > 0:
                stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_shorts])
        return stop_actions

    # 获取当前活跃的多空仓位
    # active_longs: 当前活跃的多仓列表
    # active_shorts: 当前活跃的空仓列表
    def get_active_executors_by_side(self, connector_name: str, trading_pair: str):
        active_executors_by_connector_pair = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: e.connector_name == connector_name and e.trading_pair == trading_pair and e.is_active
        )
        active_longs = [e for e in active_executors_by_connector_pair if e.side == TradeType.BUY]
        active_shorts = [e for e in active_executors_by_connector_pair if e.side == TradeType.SELL]
        return active_longs, active_shorts

    # 获取交易信号（从配置的K线数据源获取RSI信号）
    # 信号值: 1表示做多信号，-1表示做空信号，None表示无信号
    def get_signal(self, connector_name: str, trading_pair: str) -> Optional[float]:
        """获取交易信号
        
        Args:
            connector_name (str): 交易所名称
            trading_pair (str): 交易对
            
        Returns:
            Optional[float]: 
                1: RSI低于下界，产生做多信号
                -1: RSI高于上界，产生做空信号
                0: RSI在区间内，无信号
                None: 无法获取信号
        """
        # 获取K线数据，多获取10根用于计算指标
        candles = self.market_data_provider.get_candles_df(connector_name,
                                                           trading_pair,
                                                           self.config.candles_interval,
                                                           self.config.candles_length + 10)
        # 计算RSI指标
        candles.ta.rsi(length=self.config.candles_length, append=True)
        # 初始化信号列，默认为0（无信号）
        candles["signal"] = 0
        # 记录当前RSI值
        self.current_rsi = candles.iloc[-1][f"RSI_{self.config.candles_length}"]
        # RSI低于下界，设置做多信号(1)
        candles.loc[candles[f"RSI_{self.config.candles_length}"] < self.config.rsi_low, "signal"] = 1
        # RSI高于上界，设置做空信号(-1)
        candles.loc[candles[f"RSI_{self.config.candles_length}"] > self.config.rsi_high, "signal"] = -1
        # 获取最新的信号
        self.current_signal = candles.iloc[-1]["signal"] if not candles.empty else None
        return self.current_signal

    def apply_initial_setting(self):
        """应用初始设置
        
        为永续合约设置：
        1. 持仓模式(单向/双向)
        2. 杠杆倍数
        """
        # 如果还未设置账户配置
        if not self.account_config_set:
            # 遍历所有连接器（交易所）
            for connector_name, connector in self.connectors.items():
                # 如果是永续合约
                if self.is_perpetual(connector_name):
                    # 设置持仓模式（单向/双向）
                    connector.set_position_mode(self.config.position_mode)
                    # 为每个交易对设置杠杆倍数
                    for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                        connector.set_leverage(trading_pair, self.config.leverage)
            # 标记账户配置已完成
            self.account_config_set = True

    def format_status(self) -> str:
        """格式化策略状态
        
        Returns:
            str: 返回当前策略状态的字符串表示，包含：
                - 当前RSI值
                - 当前交易信号
                - 活跃订单信息
                - 持仓信息
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        # Create RSI progress bar
        if self.current_rsi is not None:
            bar_length = 50
            rsi_position = int((self.current_rsi / 100) * bar_length)
            progress_bar = ["─"] * bar_length

            # Add threshold markers
            low_threshold_pos = int((self.config.rsi_low / 100) * bar_length)
            high_threshold_pos = int((self.config.rsi_high / 100) * bar_length)
            progress_bar[low_threshold_pos] = "L"
            progress_bar[high_threshold_pos] = "H"

            # Add current position marker
            if 0 <= rsi_position < bar_length:
                progress_bar[rsi_position] = "●"

            progress_bar = "".join(progress_bar)
            lines.extend([
                "",
                f"  RSI: {self.current_rsi:.2f}  (Long ≤ {self.config.rsi_low}, Short ≥ {self.config.rsi_high})",
                f"  0 {progress_bar} 100",
            ])

        try:
            orders_df = self.active_orders_df()
            lines.extend(["", "  Active Orders:"] + ["    " + line for line in orders_df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        return "\n".join(lines)
