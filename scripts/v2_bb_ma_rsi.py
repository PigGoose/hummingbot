import os
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
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

class BbMaConfig(StrategyV2ConfigBase):
    """
    布林带和移动平均线策略的配置类
    Configuration class for Bollinger Bands and Moving Average strategy
    
    策略参数配置包括：
    1. 技术指标参数:布林带周期、标准差倍数、MA周期
    2. 风险控制参数:最大持仓、最小成交量、最大波动率
    3. 市场数据参数:最小数据点、K线配置
    4. 交易执行参数:交易对、订单大小、价差等
    5. 仓位管理参数:止损、止盈、持仓时间限制
    """
    # ======= 基本配置参数 =======
    # 当前策略脚本的文件名，自动从__file__获取
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    markets: Dict[str, List[str]] = {}
    # K线数据配置列表，定义了获取哪些周期的K线数据
    candles_config: List[CandlesConfig] = Field(default=[], client_data=None)
    # 控制器配置列表，定义了哪些控制器将被使用
    controllers_config: List[str] = []
    
    # ======= 交易所和交易对配置 =======
    # 交易所设置，默认使用binance模拟交易账户
    # 可选值包括：binance, binance_paper_trade等
    exchange: str = Field(default="okx_perpetual", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Exchange where the bot will trade"))
    
    # 交易对设置，格式为 BASE-QUOTE，如 BTC-USDT
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Trading pair in which the bot will place orders"))
    
    # K线周期设置，默认为1分钟
    candles_interval: str = Field(default="1m", client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "Candle interval (e.g. 1m for 1 minute)"))
    
    # ======= 技术指标参数 =======
    # 布林带周期长度，默认20周期，必须大于0
    bb_length: int = Field(default=20, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bollinger Bands period length"))
    
    # 布林带标准差倍数，默认2.0，必须大于0
    # 用于计算上下轨：中轨 ± (bb_std * 标准差)
    bb_std: float = Field(default=2.0, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bollinger Bands standard deviation multiplier"))
    
    # 快速移动平均线周期长度，默认5周期，必须大于0
    # 用于判断短期趋势
    fast_ma_length: int = Field(default=5, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Fast Moving Average period length (e.g. 5)"))
        
    # 慢速移动平均线周期长度，默认10周期，必须大于0
    # 用于判断中期趋势
    slow_ma_length: int = Field(default=10, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Slow Moving Average period length (e.g. 10)"))
        
    # RSI指标周期，默认15
    rsi_length: int = Field(default=15, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "RSI period length (e.g. 15)"))
    
    # ======= 风险控制参数 =======
    # 最大持仓规模，以账户余额的百分比表示，默认15%
    max_position_size: Decimal = Field(default=Decimal("0.15"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximum position size as a fraction of account balance"))
    
    # 最小成交量阈值，用于确保市场流动性充足
    # 如果市场成交量低于这个阈值，策略会认为当前市场流动性不足，会暂停交易
    min_volume_threshold: Decimal = Field(default=Decimal("1.0"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum volume threshold for trading"))
    
    # 最大允许波动率，默认5%,超过此值则暂停交易
    # 100,000 - (100,000 * 0.05) = 95,000
    # 100,000 + (100,000 * 0.05) = 105,000
    max_volatility: Decimal = Field(default=Decimal("0.05"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Maximum allowed volatility (e.g. 0.05 for 5%)"))
    
    # ======= 市场状态参数 =======
    # 策略启动所需的最小数据点数量，用于确保有足够的历史数据计算指标
    min_data_points: int = Field(default=100, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum required data points for strategy to start"))
    
    # ======= 订单执行参数 =======
    # 单个订单的数量，以基础资产计价（如对于BTC-USDT，则以BTC计价）
    order_amount: Decimal = Field(default=0.01, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Order amount (denominated in base asset)"))
    
    # 买单价格相对于参考价格的偏移百分比，默认0.1%
    # 例如：如果参考价为100，bid_spread为0.001，则买单价格为99.9
    bid_spread: Decimal = Field(default=0.001, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Bid order spread (in percent)"))
    
    # 卖单价格相对于参考价格的偏移百分比，默认0.1%
    # 例如：如果参考价为100，ask_spread为0.001，则卖单价格为100.1
    ask_spread: Decimal = Field(default=0.001, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Ask order spread (in percent)"))
    
    # 价格类型选择：
    # - mid: 使用买一价和卖一价的中间价
    # - last: 使用最新成交价
    price_type: str = Field(default="mid", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Price type to use (mid or last)"))

    # ======= 仓位管理参数 =======
    # 止损百分比，默认2%
    # 当仓位亏损超过此比例时平仓
    stop_loss: Decimal = Field(default=Decimal("0.02"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Stop loss percentage (e.g. 0.02 for 2%)"))
    
    # 止盈百分比，默认4%
    # 当仓位盈利超过此比例时平仓
    take_profit: Decimal = Field(default=Decimal("0.04"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Take profit percentage (e.g. 0.04 for 4%)"))
    
    # 持仓时间限制（秒），默认1小时
    # 当仓位持有时间超过此值时强制平仓
    time_limit: int = Field(default=60 * 60, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Position time limit in seconds (e.g. 3600 for 1 hour)"))

    # ======= 权益管理参数 =======
    # 杠杆倍数设置，默认1倍（现货）
    # 合约交易可设置更高的杠杆，如2表示2倍杠杆
    leverage: int = Field(default=1, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Leverage (e.g. 1 for spot, 2 for 2x leverage)"))
    
    # 最小价差设置，默认0.2%
    # 用于控制订单执行时的最小价差要求
    min_spread: Decimal = Field(default=Decimal("0.002"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Minimum spread percentage (e.g. 0.002 for 0.2%)"))
    
    # 仓位模式设置：
    # - ONEWAY: 单向模式，同一时间只能持有多头或空头
    # - HEDGE: 对冲模式，可同时持有多头和空头
    position_mode: PositionMode = Field(default="ONEWAY", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "Position mode (ONEWAY/HEDGE)"))
    
    @validator('position_mode', pre=True, allow_reuse=True)
    def validate_position_mode(cls, v: str) -> PositionMode:
        """
        仓位模式验证器
        
        Args:
            v (str): 输入的仓位模式值
        
        Returns:
            PositionMode: 验证后的仓位模式枚举值
        
        Raises:
            ValueError: 当输入的值不是有效的仓位模式时抛出异常
        """
        # 如果已经是 PositionMode 枚举类型，直接返回
        if isinstance(v, PositionMode):
            return v
        # 如果是字符串，转换为大写并检查是否是有效的枚举值
        if v.upper() in PositionMode.__members__:
            return PositionMode[v.upper()]
        # 如果不是有效值，抛出异常
        raise ValueError(f"Invalid position mode: {v}. Valid options are: {', '.join(PositionMode.__members__)}")
    
    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        """
        三重障碍仓位管理配置
        
        返回一个包含以下设置的TripleBarrierConfig对象:
        1. 止损止盈设置：使用配置的止损和止盈百分比
        2. 时间限制：使用配置的持仓时间限制
        3. 订单类型设置：
           - 开仓：限价单，为了更好的成交价格
           - 止盈：限价单，确保以盈利价格成交
           - 止损和时间限制：市价单，确保能快速清仓
        当前配置:
            止损:市价单,2%
            止盈:限价单,4%
            时间限制:市价单,1小时
        Returns:
            TripleBarrierConfig: 三重障碍配置对象
        """
        return TripleBarrierConfig(
            # 止损止盈设置
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            # 订单类型设置
            open_order_type=OrderType.LIMIT,      # 开仓使用限价单
            take_profit_order_type=OrderType.LIMIT,  # 止盈使用限价单
            stop_loss_order_type=OrderType.MARKET,   # 止损使用市价单
            time_limit_order_type=OrderType.MARKET   # 时间限制平仓使用市价单
        )

class BbMa(StrategyV2Base):    
     
    """
    布林带和移动平均线策略的实现类
    Implementation class for Bollinger Bands and Moving Average strategy
    
    策略逻辑：
    1. 使用布林带判断价格波动区间：
       - 上轨 = MA + n倍标准差,表示超买区域
       - 下轨 = MA - n倍标准差,表示超卖区域
       - 中轨 = 移动平均线，表示价格中枢
    
    2. 使用移动平均线判断趋势：
       - 价格在MA之上,认为趋势向上
       - 价格在MA之下,认为趋势向下
    
    3. 交易信号生成：
       - 买入条件：价格接近下轨 + MA趋势向上 + 趋势强度足够
       - 卖出条件：价格接近上轨 + MA趋势向下 + 趋势强度足够
    
    4. 风险管理：
       - 持仓规模限制
       - 波动率监控
       - 流动性检查
       - 三重障碍止损止盈
    """
    
    # 记录账户配置是否已设置
    account_config_set = False
    
    @classmethod
    def init_markets(cls, config: BbMaConfig):
        """
        初始化交易市场
        
        Args:
            config (BbMaConfig): 策略配置对象
        """
        # 设置交易市场为配置中指定的交易所和交易对
        cls.markets = {config.exchange: {config.trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: BbMaConfig):
        """
        初始化策略实例
        
        Args:
            connectors (Dict[str, ConnectorBase]): 交易所连接器字典
            config (BbMaConfig): 策略配置对象
        """
        self.logger().info(f"初始化布林带MA策略 - 交易对: {config.trading_pair}, 交易所: {config.exchange}")
        self.logger().info(f"策略参数 - BB周期: {config.bb_length}, BB标准差: {config.bb_std}, 快速MA周期: {config.fast_ma_length}, 慢速MA周期: {config.slow_ma_length}")
        self.logger().info(f"交易参数 - 最大持仓: {config.max_position_size}, 订单量: {config.order_amount}, 最小成交量: {config.min_volume_threshold}")
        self.logger().info(f"风控参数 - 止损: {config.stop_loss}, 止盈: {config.take_profit}, 持仓时限: {config.time_limit}秒")
        # 如果没有K线配置，添加默认的1分钟K线配置
        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.exchange,
                trading_pair=config.trading_pair,
                interval=config.candles_interval,  # 1分钟K线
                max_records=config.min_data_points  # 最大记录数
            ))
        # 调用父类初始化
        super().__init__(connectors, config)
        self.config = config
        self.ready = False  # 策略就绪状态
        
        # ======= 市场数据变量 =======
        # 布林带指标数据
        self.bb_upper = None   # 布林带上轨
        self.bb_middle = None  # 布林带中轨
        self.bb_lower = None   # 布林带下轨
        self.fast_ma = None    # 快速移动平均线
        self.slow_ma = None    # 慢速移动平均线
        self.prev_fast_ma = None  # 上一周期快线
        self.prev_slow_ma = None  # 上一周期慢线
        
        # 实时市场数据
        self.current_price = None      # 当前价格
        self.current_volume = None      # 当前成交量
        self.current_volatility = None  # 当前波动率
        self.current_rsi = None        # 当前 RSI 值
        
        # ======= 订单管理变量 =======
        self.buy_order = None   # 当前买单
        self.sell_order = None  # 当前卖单
        self._last_timestamp = 0  # 上次更新时间戳
        
        # ======= 性能跟踪指标 =======
        self.metrics = { 
            # 技术指标数据
            "bb_upper": None,           # 布林带上轨
            "bb_lower": None,           # 布林带下轨
            "fast_ma": None,            # 快速移动平均线
            "slow_ma": None,            # 慢速移动平均线
            "rsi": None,               # RSI指标
            
            # 市场数据
            "current_price": None,      # 当前价格
            "current_volume": None,      # 当前成交量
            "current_volatility": None,  # 当前波动率
            
            # 仓位和盈亏数据
            "position_value": Decimal("0"),  # 持仓价值
            "pnl": Decimal("0"),            # 总盈亏
            "win_count": 0,                 # 盈利交易次数
            "loss_count": 0,                # 亏损交易次数
        }
    
    def start(self, clock: Clock, timestamp: float) -> None:
        """
        策略启动方法，在策略开始运行时调用
        
        Args:
            clock (Clock): Hummingbot的时钟对象，用于时间同步
            timestamp (float): 当前时间戳
        """
        # 记录启动时间戳
        self._last_timestamp = timestamp
        # 应用初始设置
        self.apply_initial_setting()
    
    def apply_initial_setting(self):
        """
        应用初始设置，为每个合约交易所设置杠杆和仓位模式
        """
        if not self.account_config_set:
            for connector_name, connector in self.connectors.items():
                if self.is_perpetual(connector_name):
                    try:
                        # 设置仓位模式
                        connector.set_position_mode(self.config.position_mode)
                        self.logger().info(f"Set position mode to {self.config.position_mode} for {connector_name}")
                        
                        # 为每个交易对设置杠杆
                        for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                            connector.set_leverage(trading_pair, self.config.leverage)
                            self.logger().info(f"Set leverage to {self.config.leverage}x for {trading_pair} on {connector_name}")
                    except Exception as e:
                        self.logger().error(f"Error setting up {connector_name}: {str(e)}")
                        continue
            
            self.account_config_set = True
            self.logger().info("初始化账户配置完成")
    
    def on_tick(self):
        """
        每个tick周期执行的操作
        主要任务：
        1. 计算技术指标
        2. 检查交易信号
        3. 执行交易
        """
        try:
            # 计算技术指标
            self.calculate_indicators()
            
            # 如果策略未就绪，直接返回
            if not self.ready:
                self.logger().info("策略未就绪,跳过本次tick")
                return
                
            # 创建交易动作建议
            actions = self.create_actions_proposal()
            
            # 如果有交易动作，执行它们
            if actions:
                self.logger().info(f"执行交易动作: {len(actions)}个")
                for action in actions:
                    self.executors.append(action)
                    
        except Exception as e:
            self.logger().error(f"Tick处理出错: {str(e)}")
            import traceback
            self.logger().error(traceback.format_exc())
    
    def get_price_type(self) -> PriceType:
        """
        获取价格类型
        
        Returns:
            PriceType: 根据配置返回价格类型
            - 如果配置为'mid'，返回中间价
            - 否则返回最新成交价
        """
        if self.config.price_type == "mid":
            return PriceType.MidPrice  # 中间价 = (买一价 + 卖一价) / 2
        return PriceType.LastTrade    # 最新成交价
     
    def calculate_indicators(self):
        """
        计算技术指标和市场数据
        
        计算过程：
        1. 获取K线数据
        2. 计算布林带指标（上轨、中轨、下轨）
        3. 计算移动平均线
        4. 计算市场数据（成交量、波动率、RSI）
        5. 更新性能指标
        
        如果计算过程出错，策略将设置为未就绪状态。
        """
        # import traceback
        # self.logger().info(f"开始计算技术指标... 调用栈: {traceback.extract_stack()[-2]}")
        try:
            # 1. 获取K线数据
            candles_df = self.market_data_provider.get_candles_df(
                self.config.exchange,
                self.config.trading_pair,
                self.config.candles_interval
            )
            self.logger().info(f"获取到K线数据: {len(candles_df)}条")
            
            if len(candles_df) < self.config.min_data_points:
                self.logger().warning(f"数据点数不足: {len(candles_df)} < {self.config.min_data_points}")
                self.ready = False
                return
            
            # 2. 计算布林带指标
            # 计算移动平均和标准差
            ma = candles_df['close'].rolling(window=self.config.bb_length).mean()
            std = candles_df['close'].rolling(window=self.config.bb_length).std()
            
            # 计算布林带三条线
            self.bb_middle = ma.iloc[-1]  # 中轨 = MA
            self.bb_upper = ma.iloc[-1] + self.config.bb_std * std.iloc[-1]  # 上轨 = MA + n倍标准差
            self.bb_lower = ma.iloc[-1] - self.config.bb_std * std.iloc[-1]  # 下轨 = MA - n倍标准差
            
            # 3. 计算双MA
            # 计算快线和慢线
            fast_ma_series = candles_df['close'].rolling(window=self.config.fast_ma_length).mean()
            slow_ma_series = candles_df['close'].rolling(window=self.config.slow_ma_length).mean()
            
            # 保存上一周期的值
            self.prev_fast_ma = fast_ma_series.iloc[-2]
            self.prev_slow_ma = slow_ma_series.iloc[-2]
            
            # 保存当前周期的值
            self.fast_ma = fast_ma_series.iloc[-1]
            self.slow_ma = slow_ma_series.iloc[-1]
            
            # 4. 计算市场数据
            # 计算RSI (使用pandas_ta)
            self.current_rsi = candles_df.ta.rsi(close='close', length=self.config.rsi_length).iloc[-1]
            
            # 成交量
            self.current_volume = candles_df['volume'].iloc[-1]
            # 波动率（价格变化率的标准差）
            # 计算波动率，使用与布林带相同的周期
            price_changes = candles_df['close'].pct_change()
            self.current_volatility = price_changes.rolling(window=self.config.bb_length).std().iloc[-1]
            
            # 获取当前价格（根据配置的价格类型）
            self.current_price = float(self.market_data_provider.get_price_by_type(
                self.config.exchange,
                self.config.trading_pair,
                self.get_price_type()))
            
            # 5. 更新性能指标
            self.update_performance_metrics()
            
            # 记录RSI指标
            self.metrics["rsi"] = float(self.current_rsi)
            
            # 标记策略为就绪状态
            self.ready = True
            
        except Exception as e:
            # 如果计算过程出错，记录错误并设置为未就绪
            self.logger().error(f"Error calculating indicators: {str(e)}")
            self.ready = False
            
    def update_performance_metrics(self):
        """
        更新策略的性能指标
        
        更新以下指标：
        1. 技术指标值（布林带、MA）
        2. 市场数据（价格、成交量、波动率）
        
        注意：所有指标都会转换为 float 类型存储
        """
        self.logger().info("更新性能指标...")
        
        # 更新技术指标
        self.metrics["bb_upper"] = float(self.bb_upper) if self.bb_upper is not None else None
        self.metrics["bb_lower"] = float(self.bb_lower) if self.bb_lower is not None else None
        self.metrics["fast_ma"] = float(self.fast_ma) if self.fast_ma is not None else None
        self.metrics["slow_ma"] = float(self.slow_ma) if self.slow_ma is not None else None
        
        # 更新市场数据
        self.metrics["current_price"] = float(self.current_price) if self.current_price is not None else None
        self.metrics["current_volume"] = float(self.current_volume) if self.current_volume is not None else None
        self.metrics["current_volatility"] = float(self.current_volatility) if self.current_volatility is not None else None
        
        # 更新仓位和盈亏数据
        position_value = self._calculate_position_value()
        self.metrics["position_value"] = float(position_value)
        
        self.logger().info(f"当前指标 - 价格: {self.metrics['current_price']}, BB上轨: {self.metrics['bb_upper']}, BB下轨: {self.metrics['bb_lower']}, 快速MA: {self.metrics['fast_ma']}, 慢速MA: {self.metrics['slow_ma']}")
        self.logger().info(f"市场数据 - 成交量: {self.metrics['current_volume']}, 波动率: {self.metrics['current_volatility']}, 持仓价值: {self.metrics['position_value']}")
        self.metrics.update({
            # 技术指标值
            "bb_upper": float(self.bb_upper) if self.bb_upper is not None else None,  # 布林带上轨
            "bb_lower": float(self.bb_lower) if self.bb_lower is not None else None,  # 布林带下轨
            "fast_ma": float(self.fast_ma) if self.fast_ma is not None else None,  # 快速移动平均线
            "slow_ma": float(self.slow_ma) if self.slow_ma is not None else None,  # 慢速移动平均线
            
            # 市场数据
            "current_price": float(self.current_price) if self.current_price is not None else None,  # 当前价格
            "current_volume": float(self.current_volume) if self.current_volume is not None else None,  # 当前成交量
            "current_volatility": float(self.current_volatility) if self.current_volatility is not None else None,  # 当前波动率
        })
        
  
        """
        获取最大允许的持仓价值
        
        计算方法：
        1. 获取计价货币的可用余额（如USDT）
        2. 将余额乘以最大持仓比例
        
        Returns:
            Decimal: 最大允许的持仓价值
        """
        # 获取交易所连接器
        connector = self.connectors[self.config.exchange]
        # 获取计价货币余额（如BTC-USDT中的USDT）
        quote_balance = connector.get_available_balance(self.config.trading_pair.split('-')[1])
        # 计算最大持仓价值 = 余额 * 最大持仓比例
        return quote_balance * self.config.max_position_size
    
    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        """
        创建交易动作建议
        
        根据当前市场状态和交易信号，创建买入或卖出的交易动作。
        每个交易动作都包含：
        1. 交易方向（买入/卖出）
        2. 交易数量
        3. 三重障碍设置（止损、止盈、时间限制）
        
        Returns:
            List[CreateExecutorAction]: 交易动作列表
        """
        self.logger().info("开始生成交易动作...")
        
        # 检查买入信号
        if self.should_buy():
            self.logger().info(f"生成买入动作 - 数量: {self.config.order_amount}, 价格类型: {self.config.price_type}")
            return [self._create_buy_proposal()]
        
        # 检查卖出信号
        if self.should_sell():
            self.logger().info(f"生成卖出动作 - 数量: {self.config.order_amount}, 价格类型: {self.config.price_type}")
            return [self._create_sell_proposal()]
            
        self.logger().info("没有发现交易机会")
        return []
    
    def should_buy(self) -> bool:
        """
        判断当前是否应该发出买入信号
        
        买入条件：
        1. 市场状态就绪
        2. 风险指标在允许范围内
        3. 价格接近布林带下轨（范围在1%内）
        4. RSI处于超卖区（<30）
        5. 出现金叉（快线突破慢线）
        6. 趋势强度足够（避免假突破）
        
        Returns:
            bool: 如果满足所有买入条件返回 True，否则返回 False
        """
        self.logger().info(f"检查买入条件 - 当前价格: {self.current_price}, BB下轨: {self.bb_lower}, 快速MA: {self.fast_ma}, 慢速MA: {self.slow_ma}")
        # 1. 检查市场状态
        if not self._check_market_ready():
            return False
            
        # 2. 检查风险指标
        if not self._check_risk_limits():
            return False
        
        # 3. 检查价格是否接近布林带下轨
        # 定义接近下轨的阈值，比如价格在下轨1%范围内
        near_lower_band_threshold = Decimal('0.01')  # 1%
        # 使用pandas_ta计算布林带
        df = pd.DataFrame({'close': self.price_samples})
        bb = df.ta.bbands(length=self.bb_length, std=self.bb_std)
        
        # 获取最新的布林带值
        bb_lower = Decimal(str(bb['BBL_20_2.0'].iloc[-1]))  # 下轨
        
        # 计算当前价格与下轨的距离
        price_diff_from_lower = (Decimal(str(self.current_price)) - bb_lower) / bb_lower
        near_lower_band = abs(price_diff_from_lower) <= near_lower_band_threshold
        
        # 4. 检查RSI是否处于超卖区
        rsi_oversold = self.current_rsi < 30
        
        # 5. 检查是否出现金叉
        # 金叉条件：上一周期快线在慢线下方，当前周期快线突破到慢线上方
        golden_cross = (
            self.prev_fast_ma < self.prev_slow_ma and  # 上一周期快线在慢线下方
            self.fast_ma > self.slow_ma  # 当前周期快线突破到慢线上方
        )
        
        # 6. 计算并检查趋势强度（避免假突破）
        # 趋势强度 = |快线 - 慢线| / 慢线
        trend_strength = Decimal(str(abs(self.fast_ma - self.slow_ma))) / Decimal(str(self.slow_ma))
        strong_trend = trend_strength > self.config.min_spread
        
        # 记录买入信号的详细信息
        if golden_cross and strong_trend and rsi_oversold and near_lower_band:
            self.logger().info(f"Buy signal triggered: Golden cross detected (Fast MA: {self.fast_ma:.4f} > Slow MA: {self.slow_ma:.4f}), "
                           f"Trend strength: {trend_strength:.4f}, RSI: {self.current_rsi:.2f}")
            return True
        
        # 记录未触发买入的原因
        if not golden_cross:
            self.logger().debug(f"No golden cross (Fast MA: {self.fast_ma:.4f}, Slow MA: {self.slow_ma:.4f})")
        elif not strong_trend:
            self.logger().debug(f"Trend not strong enough: {trend_strength:.4f}")
        elif not rsi_oversold:
            self.logger().debug(f"RSI not in oversold territory: {self.current_rsi:.2f}")
        
        return False
    
    def should_sell(self) -> bool:
        """
        判断当前是否应该发出卖出信号
        
        卖出条件：
        1. 市场状态就绪
        2. 风险指标在允许范围内
        3. 价格接近布林带上轨（范围在1%内）
        4. RSI处于超买区（>70）
        5. 出现死叉（快线跌破慢线）
        6. 趋势强度足够（避免假突破）
        
        Returns:
            bool: 如果满足所有卖出条件返回 True，否则返回 False
        """
        self.logger().info(f"检查卖出条件 - 当前价格: {self.current_price}, BB上轨: {self.bb_upper}, 快速MA: {self.fast_ma}, 慢速MA: {self.slow_ma}")
        # 1. 检查市场状态
        if not self._check_market_ready():
            return False
            
        # 2. 检查风险指标
        if not self._check_risk_limits():
            return False
        
        # 3. 检查价格是否接近布林带上轨
        # 定义接近上轨的阈值，比如价格在上轨1%范围内
        near_upper_band_threshold = Decimal('0.01')  # 1%
        # 使用pandas_ta计算布林带
        df = pd.DataFrame({'close': self.price_samples})
        bb = df.ta.bbands(length=self.bb_length, std=self.bb_std)
        
        # 获取最新的布林带上轨值
        bb_upper = Decimal(str(bb['BBU_20_2.0'].iloc[-1]))  # 上轨
        
        # 计算当前价格与上轨的距离
        price_diff_from_upper = (Decimal(str(self.current_price)) - bb_upper) / bb_upper
        near_upper_band = abs(price_diff_from_upper) <= near_upper_band_threshold
        
        # 4. 检查RSI是否处于超买区
        rsi_overbought = self.current_rsi > 70
        
        # 5. 检查是否出现死叉
        # 死叉条件：上一周期快线在慢线上方，当前周期快线突破到慢线下方
        death_cross = (
            self.prev_fast_ma > self.prev_slow_ma and  # 上一周期快线在慢线上方
            self.fast_ma < self.slow_ma  # 当前周期快线突破到慢线下方
        )
        
        # 6. 计算并检查趋势强度（避免假突破）
        # 趋势强度 = |快线 - 慢线| / 慢线
        trend_strength = Decimal(str(abs(self.fast_ma - self.slow_ma))) / Decimal(str(self.slow_ma))
        strong_trend = trend_strength > self.config.min_spread
        
        # 记录卖出信号的详细信息
        if death_cross and strong_trend and rsi_overbought and near_upper_band:
            self.logger().info(f"Sell signal triggered: Death cross detected (Fast MA: {self.fast_ma:.4f} < Slow MA: {self.slow_ma:.4f}), "
                           f"Trend strength: {trend_strength:.4f}, RSI: {self.current_rsi:.2f}")
            return True
        
        # 记录未触发卖出的原因
        if not death_cross:
            self.logger().debug(f"No death cross (Fast MA: {self.fast_ma:.4f}, Slow MA: {self.slow_ma:.4f})")
        elif not strong_trend:
            self.logger().debug(f"Trend not strong enough: {trend_strength:.4f}")
        elif not rsi_overbought:
            self.logger().debug(f"RSI not in overbought territory: {self.current_rsi:.2f}")
        
        return False
    
    def get_spread(self) -> Decimal:
        """
        计算买卖价差
        
        计算方法：
        1. 取配置的买单价差和卖单价差的平均值
        2. 确保价差不小于最小价差要求
        
        Returns:
            Decimal: 计算得到的价差百分比
        """
        return max(
            self.config.min_spread,  # 最小价差要求（0.2%）
            (self.config.bid_spread + self.config.ask_spread) / Decimal("2")  # 买卖价差的平均值
        )
        
    def _check_market_ready(self) -> bool:
        """
        检查市场状态是否就绪
        
        检查以下条件：
        1. 策略是否就绪
        2. 市场数据提供器是否就绪
        3. 必要的市场数据是否存在（价格、成交量、波动率）
        
        Returns:
            bool: 如果所有条件都满足返回 True，否则返回 False
        """
        self.logger().info(f"检查市场状态 - 策略就绪: {self.ready}, 价格: {self.current_price is not None}, 成交量: {self.current_volume is not None}, 波动率: {self.current_volatility is not None}")
        return (
            self.ready  # 策略就绪
            and self.market_data_provider.ready  # 数据提供器就绪
            and self.current_price is not None  # 当前价格存在
            and self.current_volume is not None  # 当前成交量存在
            and self.current_volatility is not None  # 当前波动率存在
        )
    
    def _check_risk_limits(self) -> bool:
        """
        检查风险限制
        
        检查以下风险指标：
        1. 市场状态是否就绪
        2. 当前波动率是否超过限制
        3. 当前成交量是否足够
        4. 当前持仓是否超过限制
        
        Returns:
            bool: 如果所有风险指标都在允许范围内返回 True，否则返回 False
        """
        self.logger().info("开始检查风险限制...")
        
        # 1. 检查市场状态
        if not self._check_market_ready():
            self.logger().warning("市场状态未就绪")
            return False
            
        # 2. 检查波动率
        if self.current_volatility > self.config.max_volatility:
            self.logger().warning(f"当前波动率({self.current_volatility})超过限制({self.config.max_volatility})")
            return False
            
        # 3. 检查成交量
        if self.current_volume < self.config.min_volume_threshold:
            self.logger().warning(f"当前成交量({self.current_volume})低于阈值({self.config.min_volume_threshold})")
            return False
            
        # 4. 检查持仓规模
        position_value = self._calculate_position_value()
        max_position_value = self._get_max_position_value()
        if position_value > max_position_value:
            self.logger().warning(f"当前持仓({position_value})超过限制({max_position_value})")
            return False
            
        self.logger().info("风险检查通过")
        return True
    
    def _calculate_position_value(self) -> Decimal:
        """
        计算当前持仓价值
        
        计算方法：
        1. 获取基础资产的可用余额（如BTC）
        2. 将余额乘以当前价格得到价值
        
        Returns:
            Decimal: 当前持仓的价值（以计价货币计价）
        """
        # 获取交易所连接器
        connector = self.connectors[self.config.exchange]
        # 获取基础资产余额（如BTC-USDT中的BTC）
        base_balance = connector.get_available_balance(self.config.trading_pair.split('-')[0])
        # 计算总价值 = 余额 * 当前价格
        return base_balance * Decimal(str(self.current_price))
        
    def _get_max_position_value(self) -> Decimal:
        """
        获取最大允许的持仓价值
        
        计算方法：
        1. 获取计价货币的可用余额（如USDT）
        2. 将余额乘以最大持仓比例
        
        Returns:
            Decimal: 最大允许的持仓价值
        """
        # 获取交易所连接器
        connector = self.connectors[self.config.exchange]
        
        # 获取计价货币余额（如BTC-USDT中的USDT）
        quote_balance = connector.get_available_balance(self.config.trading_pair.split('-')[1])
        
        # 计算最大持仓价值 = 余额 * 最大持仓比例
        max_position = quote_balance * self.config.max_position_size
        
        self.logger().info(f"计算最大持仓 - 可用余额: {quote_balance}, 最大持仓比例: {self.config.max_position_size}, 最大持仓价值: {max_position}")
        
        return max_position
        
    def _create_buy_proposal(self) -> CreateExecutorAction:
        """
        创建买入订单建议
        
        创建一个包含以下内容的买入订单建议：
        1. 买入价格：当前价格减去买入价差
        2. 买入数量：配置的订单数量
        3. 三重障碍设置：止损、止盈、时间限制
        
        Returns:
            CreateExecutorAction: 买入订单执行动作
        """
        # 获取当前市场价格
        current_price = Decimal(str(self.current_price))
        
        # 计算买入价格 = 当前价格 * (1 - 买入价差)
        buy_price = current_price * (Decimal('1') - self.config.bid_spread)
        
        # 创建仓位执行器配置
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),              # 当前时间戳
            trading_pair=self.config.trading_pair,                   # 交易对
            exchange=self.config.exchange,                           # 交易所
            side=TradeType.BUY,                                      # 买入方向
            entry_price=buy_price,                                   # 买入价格
            amount=self.config.order_amount,                         # 买入数量
            position_mode=self.config.position_mode,                 # 仓位模式
            leverage=self.config.leverage,                           # 杠杆倍数
            entry_order_type=OrderType.LIMIT,                        # 限价单
            triple_barrier_conf=self.config.triple_barrier_config()  # 三重障碍配置
        )
        
        # 创建执行动作
        action = CreateExecutorAction(
            executor_config=executor_config
        )
        
        self.logger().info(
            f"创建买入订单 - 交易对: {self.config.trading_pair}, "
            f"价格: {buy_price}, 数量: {self.config.order_amount}, "
            f"止损: {self.config.stop_loss}, 止盈: {self.config.take_profit}, "
            f"时间限制: {self.config.time_limit}秒"
        )
        
        return action
        
    def _create_sell_proposal(self) -> CreateExecutorAction:
        """
        创建卖出订单建议
        
        创建一个包含以下内容的卖出订单建议：
        1. 卖出价格：当前价格加上卖出价差
        2. 卖出数量：配置的订单数量
        3. 三重障碍设置：止损、止盈、时间限制
        
        Returns:
            CreateExecutorAction: 卖出订单执行动作
        """
        # 获取当前市场价格
        current_price = Decimal(str(self.current_price))
        
        # 计算卖出价格 = 当前价格 * (1 + 卖出价差)
        sell_price = current_price * (Decimal('1') + self.config.ask_spread)
        
        # 创建仓位执行器配置
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),              # 当前时间戳
            trading_pair=self.config.trading_pair,                   # 交易对
            exchange=self.config.exchange,                           # 交易所
            side=TradeType.SELL,                                     # 卖出方向
            entry_price=sell_price,                                  # 卖出价格
            amount=self.config.order_amount,                         # 卖出数量
            position_mode=self.config.position_mode,                 # 仓位模式
            leverage=self.config.leverage,                           # 杠杆倍数
            entry_order_type=OrderType.LIMIT,                        # 限价单
            triple_barrier_conf=self.config.triple_barrier_config()  # 三重障碍配置
        )
        
        # 创建执行动作
        action = CreateExecutorAction(
            executor_config=executor_config
        )
        
        self.logger().info(
            f"创建卖出订单 - 交易对: {self.config.trading_pair}, "
            f"价格: {sell_price}, 数量: {self.config.order_amount}, "
            f"止损: {self.config.stop_loss}, 止盈: {self.config.take_profit}, "
            f"时间限制: {self.config.time_limit}秒"
        )
        
        return action
        
    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        """
        创建停止交易动作建议
        
        在以下情况下会触发停止动作：
        1. 市场状态异常（流动性不足、波动率过高）
        2. 风险指标超限
        3. 策略停止信号
        4. 紧急情况处理
        
        Returns:
            List[StopExecutorAction]: 停止动作列表，包含需要停止的执行器ID
        """
        stop_actions = []
        
        try:
            # 检查是否需要停止所有执行器
            should_stop_all = (
                not self._check_market_ready() or  # 市场状态未就绪
                self.current_volatility > self.config.max_volatility or  # 波动率过高
                self.current_volume < self.config.min_volume_threshold  # 流动性不足
            )
            
            if should_stop_all:
                self.logger().warning("检测到异常市场状态，停止所有交易执行器")
                # 获取所有活跃的执行器
                active_executors = [executor for executor in self.executors if not executor.is_closed]
                
                # 为每个活跃执行器创建停止动作
                for executor in active_executors:
                    stop_action = StopExecutorAction(
                        executor_id=executor.id,
                        timestamp=self.market_data_provider.time()
                    )
                    stop_actions.append(stop_action)
                    self.logger().info(f"创建停止动作 - 执行器ID: {executor.id}")
            
            # 记录停止动作数量
            if stop_actions:
                self.logger().info(f"总计创建 {len(stop_actions)} 个停止动作")
            
        except Exception as e:
            self.logger().error(f"创建停止动作时发生错误: {str(e)}")
            import traceback
            self.logger().error(traceback.format_exc())
        
        return stop_actions
    