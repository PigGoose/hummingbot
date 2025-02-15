import os
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType, PositionSide
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction, StopExecutorAction

class BbEmaRsiConfig(StrategyV2ConfigBase):
    """
    布林带和EMA策略的配置类
    Configuration class for Bollinger Bands and EMA strategy
    
    策略参数配置包括：
    1. 技术指标参数:布林带周期、标准差倍数、EMA5周期、EMA10周期
    2. 风险控制参数:最大持仓、订单量、最小成交量
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
    controllers_config: List[str] = Field(default=["main"], client_data=None)
    
    # ======= 交易所和交易对配置 =======
    # 交易所设置，默认使用binance模拟交易账户
    # 可选值包括：binance, binance_paper_trade等
    connector_name: str = Field(default="okx_perpetual", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "机器人将在哪个交易所进行交易"))
    
    # 交易对设置，格式为 BASE-QUOTE，如 BTC-USDT
    trading_pair: str = Field(default="BTC-USDT", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "机器人将交易的交易对"))
    
    # K线周期设置，默认为1分钟
    candles_interval: str = Field(default="1m", client_data=ClientFieldData(
        prompt_on_new=False, prompt=lambda mi: "K线周期（例如：1m 代表 1分钟）"))
    
    # ======= 技术指标参数 =======
    # 布林带周期长度，默认40周期，必须大于0
    bb_length: Decimal = Field(default=40, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "布林带周期长度"))
    
    # 布林带标准差倍数，默认2.0，必须大于0
    # 用于计算上下轨：中轨 ± (bb_std * 标准差)
    bb_std: Decimal = Field(default=2.0, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "布林带标准差倍数"))
   
    # 布林带中轨权重，默认3，必须大于0
    bb_middle: Decimal = Field(default=3.0, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "布林带中轨权重（默认：3.0）"))
        
    # 布林带下轨权重，默认2，必须大于0
    bb_down: Decimal = Field(default=2.0, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "布林带下轨权重（默认：2.0）"))
        
    # 布林带上轨权重，默认2，必须大于0
    bb_up: Decimal = Field(default=2.0, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "布林带上轨权重（默认：2.0）")) 
    
    # 5周期指数移动平均线周期长度，默认5周期，必须大于0
    # 用于判断短期趋势
    ema5_length: Decimal = Field(default=5, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "EMA5周期长度（默认：5）"))
        
    # 10周期指数移动平均线周期长度，默认10周期，必须大于0
    # 用于判断中期趋势
    ema10_length: Decimal = Field(default=10, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "EMA10周期长度（默认：10）"))
        
    # RSI指标周期，默认15
    rsi_length: Decimal = Field(default=15, gt=0, client_data=ClientFieldData(  
        prompt_on_new=True, prompt=lambda mi: "RSI 周期（默认：15）"))
        
    # RSI进场阈值，默认70，必须在0-100之间
    rsi_entry: Decimal = Field(default=70.0, gt=0, lt=100, client_data=ClientFieldData(        # XRH    Entry  不是这个意思  modified by ZXY at 2025-02-14 10:12
        prompt_on_new=True, prompt=lambda mi: "RSI 进场阈值 (默认: 70)"))
        
    # RSI出场阈值，默认30，必须在0-100之间
    rsi_exit: Decimal = Field(default=30.0, gt=0, lt=100, client_data=ClientFieldData(         # XRH    Exit   不是这个意思  modified by ZXY at 2025-02-14 10:12
        prompt_on_new=True, prompt=lambda mi: "RSI 出场阈值 (默认: 30)"))
        
    # 入场斜率阈值，默认15，必须大于0
    # 用于判断入场（归一化）斜率，值越大要求（归一化）斜率越强                                                           # XRH 名字叫做（归一化）斜率，Δz，不要叫做趋势强度吧，否则太糊了，下同  2025-02-14 12:20 #ZXY 修改了名字 at 2025-02-14 14:30
    slope_entry: Decimal = Field(default=15.0, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "入场（归一化）斜率阈值（默认：15.0）"))
        
    # 离场斜率阈值，默认6，必须大于0
    # 用于判断离场（归一化）斜率，值越大要求趋势越强
    slope_exit: Decimal = Field(default=6.0, gt=0, client_data=ClientFieldData(                    #默认<6   XRH  02-13 19:38  modified by ZXY at 2025-02-14 10:12
        prompt_on_new=True, prompt=lambda mi: "离场斜率阈值（默认：6.0）"))
        
    # ======= 风险控制参数 =======
    # 最大持仓规模，以账户余额的百分比表示，默认25%
    max_position_size: Decimal = Field(default=Decimal("0.25"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "最大持仓规模（默认：15%）"))
    
    # 最小成交量阈值，用于确保市场流动性充足
    # 如果市场成交量低于这个阈值，策略会认为当前市场流动性不足，会暂停交易
    '''
    代表的是每个交易周期（1分钟）内的最小成交量要求。
    如果当前成交量低于这个阈值，策略会认为市场流动性不足，从而不执行交易,
    以避免在流动性不足的市场中交易可能带来的风险（如滑点过大、成交困难等）。
    阈值的单位是基础货币（如BTC-USDT中的BTC）的数量
    remark by zxy at 2025-02-14 14:30
    '''
    min_volume_threshold: Decimal = Field(default=Decimal("0.0"), gt=0, client_data=ClientFieldData(                   # XRH 1这个值代表最小成交量为1min起码交易1次吗？ 2025-02-14 12:20
        prompt_on_new=True, prompt=lambda mi: "最小成交量阈值（默认：0.0）"))
    
    # 最大允许波动率，默认5%,超过此值则暂停交易
    # 100,000 - (100,000 * 0.05) = 95,000
    # 100,000 + (100,000 * 0.05) = 105,000
    max_volatility: Decimal = Field(default=Decimal("0.05"), gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "最大允许波动率（默认：5%)"))
    
    # ======= 市场状态参数 =======
    # 策略启动所需的最小数据点数量，用于确保有足够的历史数据计算指标
    min_data_points: Decimal = Field(default=100, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "最小数据点数量（默认：100）"))
    
    # ======= 订单执行参数 =======
    # 单个订单的数量，以基础资产计价（如对于BTC-USDT，则以BTC计价）
    order_amount: Decimal = Field(default=0.002, gt=0, client_data=ClientFieldData(                                      # XRH 默认0.002  2025-02-13 19:38 modified by ZXY at 2024-02-14 10:15
        prompt_on_new=True, prompt=lambda mi: "单个订单的数量（以基础资产计价，如对于BTC-USDT，则以BTC计价）"))
    
    # 价格类型选择：
    # - mid: 使用买一价和卖一价的中间价
    # - last: 使用最新成交价
    price_type: str = Field(default="mid", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "价格类型选择：（mid或last）"))

    # ======= 仓位管理参数 =======
    # 止损百分比，默认9%
    # 当仓位亏损超过此比例时平仓
    stop_loss: Decimal = Field(default=Decimal("0.09"), gt=0, client_data=ClientFieldData(                            # XRH 这里的止损是加杠杆还是没加？我们止损是9%（加杠杆的）02-13 19:52 # ZXY 杠杆后 at 2024-02-14 10:20
        prompt_on_new=True, prompt=lambda mi: "止损百分比（默认：9%)"))
    
    # 止盈百分比，默认100%
    # 当仓位盈利超过此比例时平仓
    take_profit: Decimal = Field(default=Decimal("1.00"), gt=0, client_data=ClientFieldData(                          # XRH 不止盈 默认100%       # 02-13 19:52 modified by ZXY at 2024-02-14 10:15
        prompt_on_new=True, prompt=lambda mi: "止盈百分比（默认：100%)"))
    
    # 持仓时间限制（秒），默认1小时
    # 当仓位持有时间超过此值时强制平仓
    time_limit: int = Field(default=60 * 60 * 2, gt=0, client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "持仓时间限制（秒）（默认：7200）"))                                       # XRH 2小时持仓 7200s （60*60*2）

    # ======= 权益管理参数 =======
    # 杠杆倍数设置，默认10倍
    leverage: int = Field(default=10, gt=0, client_data=ClientFieldData(                                            #XRH 默认10   02-13 19:52  # modified by ZXY at 2024-02-14 10:15
        prompt_on_new=True, prompt=lambda mi: "杠杆倍数（默认：10）"))
    
    # 最小价差设置，默认0.01%
    # 用于控制订单执行时的最小价差要求                                                                                      
    min_order_spread: Decimal = Field(default=Decimal("0.0001"), gt=0, client_data=ClientFieldData(                      #XRH 啥意思？ # 02-13 19:52 remark by ZXY 建仓和平仓时的价差 at 2024-02-14 10:15
        prompt_on_new=True, prompt=lambda mi: "最小价差（默认：0.01%)"))
    
    # bid_price_offset ask_price_offset 这两个参数是为了根据市场实时计算价差 目前使用的是固定值min_order_spread  # modified by ZXY at 2024-02-14 12:04
    # 买单价格相对于参考价格的偏移百分比，默认0.1%                                                                                   # XRH 0.1% 有点大了 2025-02-14 12:20 # remark by ZXY 这个暂时没用，后面考虑优化时使用 at 2024-02-14 10:15
    # 例如：如果参考价为100，bid_price_offset为0.001，则买单价格为99.9
    # bid_price_offset: Decimal = Field(default=0.001, gt=0, client_data=ClientFieldData(                                        # XRH 朝上买还是朝下买？  2025-02-13 19:38 
    #     prompt_on_new=True, prompt=lambda mi: "买单价格相对于参考价格的偏移百分比（默认：0.1%)"))
    
    # # 卖单价格相对于参考价格的偏移百分比，默认0.1%
    # # 例如：如果参考价为100，ask_price_offset为0.001，则卖单价格为100.1
    # ask_price_offset: Decimal = Field(default=0.001, gt=0, client_data=ClientFieldData(                                        # XRH 朝上买还是朝下买？  2025-02-13 19:52
    #     prompt_on_new=True, prompt=lambda mi: "卖单价格相对于参考价格的偏移百分比（默认：0.1%)"))
    
    
    # 仓位模式设置：
    # - ONEWAY: 单向模式，同一时间只能持有多头或空头
    # - HEDGE: 对冲模式，可同时持有多头和空头
    position_mode: PositionMode = Field(default="ONEWAY", client_data=ClientFieldData(
        prompt_on_new=True, prompt=lambda mi: "仓位模式（默认：ONEWAY）"))
    
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
            止损:市价单,9%                                                                        # XRH 止损9% 2025-02-14 12:20 
            止盈:限价单,100%                                                                        # XRH 止盈100% 2025-02-14 12:20
            时间限制:市价单,2小时                                                                  # XRH 2小时持仓 2025-02-14 12:20
        Returns:
            TripleBarrierConfig: 三重障碍配置对象
        """
        return TripleBarrierConfig(
            # 止损止盈设置
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            # 订单类型设置
            open_order_type=OrderType.LIMIT,         # 开仓使用限价单
            take_profit_order_type=OrderType.LIMIT,  # 止盈使用限价单
            stop_loss_order_type=OrderType.MARKET,   # 止损使用市价单
            time_limit_order_type=OrderType.MARKET   # 时间限制平仓使用市价单
        )

class BbEmaRsi(StrategyV2Base):
     
    """
    布林带和移动平均线策略的实现类
    Implementation class for Bollinger Bands and Moving Average strategy
    
    策略逻辑：
    1. 使用布林带判断价格波动区间：
       - 上轨 = EMA + n倍标准差,表示超买区域
       - 下轨 = EMA - n倍标准差,表示超卖区域
       - 中轨 = 移动平均线，表示价格中枢
    
    2. 使用移动平均线判断买卖时机：
       - EMA5处EMA10上方 趋势向上
       - EMA5处EMA10下方 趋势向下
    
    3. 交易信号生成：
       - 买入条件：价格接近下轨 + EMA趋势向上 + 达到k阈值
       - 卖出条件：价格接近上轨 + EMA趋势向下 + 达到k阈值
    
    4. 风险管理：
       - 持仓规模限制
       - 波动率监控
       - 流动性检查
       - 三重障碍止损止盈
    """
    
    # 记录账户配置是否已设置
    account_config_set = False
    
    @classmethod
    def init_markets(cls, config: BbEmaRsiConfig):
        """
        初始化交易市场
        
        Args:
            config (BbEmaRsiConfig): 策略配置对象
        """
        # 设置交易市场为配置中指定的交易所和交易对
        cls.markets = {config.connector_name: {config.trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: BbEmaRsiConfig):
        """
        初始化策略实例
        
        Args:
            connectors (Dict[str, ConnectorBase]): 交易所连接器字典
            config (BbEmaRsiConfig): 策略配置对象
        """
        # 初始化交易市场
        self.init_markets(config)
        
        # 确保控制器配置包含"main"
        if "main" not in config.controllers_config:
            config.controllers_config.append("main")
            
        self.logger().info(f"初始化布林带MA策略 - 交易对: {config.trading_pair}, 交易所: {config.connector_name}")
        self.logger().info(f"策略参数 - BB周期: {config.bb_length}, BB标准差: {config.bb_std}, EMA5周期: {config.ema5_length}, EMA10周期: {config.ema10_length}")
        self.logger().info(f"交易参数 - 最大持仓: {config.max_position_size}, 订单量: {config.order_amount}, 最小成交量: {config.min_volume_threshold}")
        self.logger().info(f"风控参数 - 止损: {config.stop_loss}, 止盈: {config.take_profit}, 持仓时限: {config.time_limit}秒")
        # 如果没有K线配置，添加默认的1分钟K线配置
        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.connector_name,
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
        self.ema5 = None    # 5周期指数移动平均线
        self.ema10 = None    # 10周期指数移动平均线
        self.prev_ema5 = None  # 上一周期EMA5
        self.prev_ema10 = None  # 上一周期EMA10
        
        # 实时市场数据
        self.current_price = None      # 当前价格
        self.current_volume = None      # 当前成交量
        self.current_volatility = None  # 当前波动率
        self.rsi = None        # 当前 RSI 值
        self.sigma_t = None            # EMA5的标准差                           
                                                                           # XRH 可以再加一个EMA5 ave？
        # ======= 时间戳管理 =======
        self._last_timestamp = 0  # 上次更新时间戳
        
        # ======= 性能跟踪指标 =======
        self.metrics = { 
            # 技术指标数据
            "bb_upper": None,            # 布林带上轨
            "bb_lower": None,            # 布林带下轨
            "ema5": None,                # 5周期指数移动平均线
            "ema10": None,               # 10周期指数移动平均线
            "rsi": None,                 # RSI指标
            
            # 市场数据
            "current_price": None,       # 当前价格
            "current_volume": None,      # 当前成交量
            "current_volatility": None,  # 当前波动率
            
            # 仓位和盈亏数据
            "position_value": Decimal("0"),  # 持仓价值
            "pnl": Decimal("0"),             # 总盈亏
            "win_count": 0,                  # 盈利交易次数
            "loss_count": 0,                 # 亏损交易次数
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
        self._apply_initial_setting()
    
    def _apply_initial_setting(self):                                                                           # XRH 可以再检查一下对不对 2025-02-14 12:20 # pass 且日志修改为中文  modified by ZXY  at 2024-02-14 14:30
        """
        应用初始设置，为每个合约交易所设置杠杆和仓位模式
        """
        if not self.account_config_set:
            for connector_name, connector in self.connectors.items():
                if self.is_perpetual(connector_name):
                    try:
                        # 设置仓位模式
                        connector.set_position_mode(self.config.position_mode)
                        self.logger().info(f"为{connector_name}设置仓位模式为{self.config.position_mode}")
                        
                        # 为每个交易对设置杠杆
                        for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                            connector.set_leverage(trading_pair, self.config.leverage)
                            self.logger().info(f"为{connector_name}的{trading_pair}设置{self.config.leverage}倍杠杆")
                    except Exception as e:
                        self.logger().error(f"设置{connector_name}时发生错误: {str(e)}")
                        continue
            
            self.account_config_set = True
            self.logger().info("账户配置初始化完成")
    
    def on_tick(self):
        """
        每个tick周期执行的操作
        主要任务：
        1. 计算技术指标
        2. 检查交易信号
        3. 执行交易
        """
        try:
            # 1. 计算技术指标
            self._calculate_indicators()
            
            # 如果策略未就绪，直接返回
            if not self.ready:
                self.logger().info("策略未就绪,跳过本次tick")
                return
            
            # 2. 检查是否需要平仓
            exit_actions = self._exit_actions_proposal()
            if exit_actions:
                self.logger().info("检测到平仓信号，执行平仓操作")
                return exit_actions
            
            # 3. 创建交易动作建议
            entry_actions = self._entry_actions_proposal()
            if entry_actions:
                self.logger().info(f"执行交易动作: {len(entry_actions)}个")
                for action in entry_actions:
                    # 通过executor_orchestrator创建执行器
                    self.executor_orchestrator.create_executor(action)
                    
        except Exception as e:
            self.logger().error(f"Tick处理出错: {str(e)}")
            import traceback
            self.logger().error(traceback.format_exc())
    
    def _get_price_type(self) -> PriceType:
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
     
    def _calculate_ema_indicators(self, candles_df) -> Optional[tuple]:
        """
        计算EMA指标
        
        Args:
            candles_df (pd.DataFrame): K线数据
            
        计算过程：
        1. 计算 EMA5 和 EMA10
        2. 保存条件蜡烛上一周期和当前周期的值
        
        Returns:
            tuple: (ema5_series, ema10_series, prev_ema5, prev_ema10, ema5, ema10)
        """
        # 计算 EMA5 和 EMA10
        self.ema5_series = candles_df.ta.ema(close='close', length=self.config.ema5_length)
        self.ema10_series = candles_df.ta.ema(close='close', length=self.config.ema10_length)
        # 保存条件蜡烛上一周期的值
        self.prev_ema5 = self.ema5_series.iloc[-3]         # 条件蜡烛的前一根K线的EMA5
        self.prev_ema10 = self.ema10_series.iloc[-3]       # 条件蜡烛的前一根K线的EMA10
        # 保存条件蜡烛当前周期的值
        self.ema5 = self.ema5_series.iloc[-2]              # 条件蜡烛的EMA5
        self.ema10 = self.ema10_series.iloc[-2]            # 条件蜡烛的EMA10
        
        self.ema_data = (self.prev_ema5, self.prev_ema10, self.ema5, self.ema10)
        
        return self.ema_data

    def _calculate_volume_and_volatility(self, candles_df) -> Optional[tuple]:  
        """
        计算成交量和波动率
        
        Args:
            candles_df (pd.DataFrame): K线数据
            
        Returns:
            tuple: (current_volume, current_volatility)
        """
        # 成交量
        self.current_volume = candles_df['volume'].iloc[-1]
        # 使用 ATR 指标计算波动率
        atr = candles_df.ta.atr(high='high', low='low', close='close', length=self.config.bb_length)
        self.current_volatility = atr.iloc[-1] / candles_df['close'].iloc[-1]  # 将ATR标准化为百分比
        
        return (self.current_volume, self.current_volatility)

    def _calculate_market_data(self, candles_df) -> Optional[tuple]:
        """
        计算市场数据（成交量、波动率、当前价格）
        
        Args:
            candles_df (pd.DataFrame): K线数据
            
        Returns:
            tuple: (current_volume, current_volatility, current_price)
        """
        # 计算当前成交量
        self.current_volume = candles_df['volume'].iloc[-1]
        
        # 使用 ATR 指标计算波动率
        atr = candles_df.ta.atr(high='high', low='low', close='close', length=self.config.bb_length)
        self.current_volatility = atr.iloc[-1] / candles_df['close'].iloc[-1]  # 将ATR标准化为百分比
        
        # 获取当前价格（根据配置的价格类型）
        self.current_price = Decimal(str(self.market_data_provider.get_price_by_type(
            self.config.connector_name,
            self.config.trading_pair,
            self._get_price_type())))
            
        return (self.current_volume, self.current_volatility, self.current_price)

    def _calculate_rsi(self, candles_df) -> Optional[Decimal]:
        """
        计算RSI指标
        
        Args:
            candles_df (pd.DataFrame): K线数据
            
        Returns:
            float: RSI值
        """
        self.rsi = candles_df.ta.rsi(close='close', length=self.config.rsi_length).iloc[-2]  # 使用条件蜡烛的RSI值
        return self.rsi

    def _calculate_normalized_slope(self, candles_df) -> Optional[Decimal]:
        """
        计算归一化斜率 ΔZ(t)
        
        Args:
            candles_df (pd.DataFrame): K线数据
            
        计算过程：
        1. 确定窗口大小
        2. 计算均值和标准差
        3. 计算z值和斜率
        
        Returns:
            tuple: (delta_z, sigma_t) 归一化斜率和标准差
        """
        window_size = min(81, len(candles_df))
        ema5_window = self.ema5_series.iloc[-window_size:]
        self.mu_t = Decimal(str(ema5_window.mean()))
        self.sigma_t = Decimal(str(ema5_window.std()))  # 保存为类变量
        z_t = (Decimal(str(self.ema5)) - self.mu_t) / self.sigma_t if self.sigma_t != 0 else Decimal('0')
        z_prev = (Decimal(str(ema5_window.iloc[-3])) - self.mu_t) / self.sigma_t if self.sigma_t != 0 else Decimal('0')
        delta_z = z_t - z_prev
        self.delta_z = delta_z
        
        return (self.delta_z, self.sigma_t)

    def _calculate_bb_bands(self, candles_df) -> Optional[tuple]:
        """
        计算布林带指标
        
        Args:
            candles_df (pd.DataFrame): K线数据
            
        计算过程：
        1. 使用pandas_ta库的bbands函数计算布林带
        2. 设置上轨、中轨、下轨值
        
        Returns:
            tuple: (bb_middle, bb_upper, bb_lower) 布林带中轨、上轨、下轨
        """
        # 使用pandas_ta库的bbands函数计算布林带
        # close: 使用收盘价
        # length: 移动平均的周期
        # std: 标准差的倍数，用于计算上下轨
        # 返回值包含三条带: BBU(上轨), BBM(中轨), BBL(下轨)
        bb = candles_df.ta.bbands(close='close', length=self.config.bb_length, std=self.config.bb_std)  # 函数文档: https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/bbands.py
        self.bb_middle = bb[f'BBM_{self.config.bb_length}_{self.config.bb_std}'].iloc[-2]  # 条件蜡烛的布林带中轨
        self.bb_upper = bb[f'BBU_{self.config.bb_length}_{self.config.bb_std}'].iloc[-2]   # 条件蜡烛的布林带上轨
        self.bb_lower = bb[f'BBL_{self.config.bb_length}_{self.config.bb_std}'].iloc[-2]   # 条件蜡烛的布林带下轨
        
        return (self.bb_middle, self.bb_upper, self.bb_lower)

    def _get_candles_data(self) -> Optional[pd.DataFrame]:
        """
        获取K线数据并检查数据点数是否满足要求
        
        Returns:
            pd.DataFrame: 满足要求的K线数据，如果数据点数不足则返回None
        """
        candles_df = self.market_data_provider.get_candles_df(
            self.config.connector_name,
            self.config.trading_pair,
            self.config.candles_interval
        )
        self.logger().info(f"获取到K线数据: {len(candles_df)}条")
        
        if len(candles_df) < self.config.min_data_points:
            self.logger().warning(f"数据点数不足: {len(candles_df)} < {self.config.min_data_points}")
            self.ready = False
            return None
            
        return candles_df

    def _calculate_indicators(self):
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
        try:
            # 1. 获取K线数据
            candles_df = self._get_candles_data()
            if candles_df is None:
                return
            
            # 2. 计算布林带指标
            self._calculate_bb_bands(candles_df)
            
            # 3. 计算EMA
            # 计算 EMA5 和 EMA10
            self._calculate_ema_indicators(candles_df)
            
            # 4. 计算归一化斜率 ΔZ(t)
            self._calculate_normalized_slope(candles_df)
            
            # 5. 计算RSI (使用pandas_ta)
            self._calculate_rsi(candles_df)
            
            # 6. 计算市场数据
            # 成交量
            # 计算市场数据（成交量、波动率、当前价格）
            self._calculate_market_data(candles_df)
            
            # 7. 更新性能指标
            self._update_performance_metrics()
            
            # 标记策略为就绪状态
            self.ready = True
            
        except Exception as e:
            # 如果计算过程出错，记录错误并设置为未就绪
            self.logger().error(f"Error calculating indicators: {str(e)}")
            self.ready = False
            
    def _update_performance_metrics(self):
        """
        更新策略的性能指标
        
        更新以下指标：
        1. 技术指标值（布林带、EMA）
        2. 市场数据（价格、成交量、波动率）
        
        注意：所有指标都会转换为 float 类型存储
        """
        self.logger().info("更新性能指标...")
        
        # 更新所有指标和市场数据
        position_value = self._calculate_position_value()
        # 优化更新方式，删除冗余代码 by ZXY at 2025-02-14 14:30
        self.metrics.update({
            # 技术指标值
            "bb_upper": float(self.bb_upper) if self.bb_upper is not None else None,  # 布林带上轨
            "bb_lower": float(self.bb_lower) if self.bb_lower is not None else None,  # 布林带下轨
            "ema5": float(self.ema5) if self.ema5 is not None else None,  # 5周期指数移动平均线
            "ema10": float(self.ema10) if self.ema10 is not None else None,  # 10周期指数移动平均线
            "rsi": float(self.rsi) if self.rsi is not None else None,  # RSI指标
            
            # 市场数据
            "current_price": float(str(self.current_price)) if self.current_price is not None else None,  # 当前价格
            "current_volume": float(str(self.current_volume)) if self.current_volume is not None else None,  # 当前成交量
            "current_volatility": float(str(self.current_volatility)) if self.current_volatility is not None else None,  # 当前波动率
            "position_value": float(position_value)  # 持仓价值
        })
        
        self.logger().info(f"当前指标 - 价格: {self.metrics['current_price']}, BB上轨: {self.metrics['bb_upper']}, BB下轨: {self.metrics['bb_lower']}, EMA5: {self.metrics['ema5']}, EMA10: {self.metrics['ema10']}, RSI: {self.metrics['rsi']}")
        self.logger().info(f"市场数据 - 成交量: {self.metrics['current_volume']}, 波动率: {self.metrics['current_volatility']}, 持仓价值: {self.metrics['position_value']}")

    def _entry_actions_proposal(self) -> List[CreateExecutorAction]:
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
        
        # 检查做多信号
        if self._check_entry_long_conditions():
            self.logger().info(f"生成做多动作 - 数量: {self.config.order_amount}, 价格类型: {self.config.price_type}")
            return [self._create_long_proposal()]
        
        # 检查做空信号
        if self._check_entry_short_conditions():
            self.logger().info(f"生成做空动作 - 数量: {self.config.order_amount}, 价格类型: {self.config.price_type}")
            return [self._create_short_proposal()]
            
        self.logger().info("没有发现交易机会")
        return []
    
    def _check_entry_long_conditions(self) -> bool:
        # 记录当前市场状态
        self.logger().info(
            f"检查做多条件 - 当前价格: {self.current_price}, "
            f"BB中轨: {self.bb_middle}, BB下轨: {self.bb_lower}, "                           # XRH 这里如果是实时的，那是不是得为.iloc[-1]? 下同  2025-02-14 12:20 # 此处是一个数据，不是一组数据 ZXY at 2025-02-14 14:30
            f"EMA5: {self.ema5}, EMA10: {self.ema10}, "
            f"RSI: {self.rsi}"
        )
        
        # 1. 检查市场状态
        if not self._check_market_ready():
            self.logger().info("市场状态未就绪，不执行做多")
            return False
            
        # 2. 检查风险指标
        if not self._check_risk_limits():
            self.logger().info("风险指标超限，不执行做多")
            return False
        
        # 3. 检查技术指标
        # 条件1：EMA5和EMA10的位置判断 + 金叉判断
        # TODO XRH审计逻辑       config.bb_middle 能读取到吗？我这里是白色，需要套Decimal吗？ 下同。 其他pass  02-13 19：10 # ZXY 可以点了 需要Decimal 保持数据一致性 by ZXY at 2024-02-14 10:40
        bb_threshold = (self.config.bb_middle * Decimal(str(self.bb_middle)) + self.config.bb_down * Decimal(str(self.bb_lower))) / Decimal('5')
        self.logger().info(f"BB中轨K: {self.bb_middle}, BB下轨: {self.bb_lower}, 阈值: {bb_threshold}")              # XRH BB中轨打印错了 应该打印self.bb_middle 而不是self.config.bb_middle 2025-02-14 12:20 # modified by zxy at 2024-02-14 14:30
        ema5_decimal = Decimal(str(self.ema5))
        ema10_decimal = Decimal(str(self.ema10))
        prev_ema5_decimal = Decimal(str(self.prev_ema5))
        prev_ema10_decimal = Decimal(str(self.prev_ema10))
        price_cond = (ema5_decimal < bb_threshold) or (ema10_decimal < bb_threshold)
        cross_cond = ((prev_ema5_decimal - prev_ema10_decimal) < Decimal('0')) and ((ema5_decimal - ema10_decimal) > Decimal('0'))     # XRH 计算减法要套括号吗？ 2025-02-14 12:20 # python - 优先级大于 < 和 > # remark by zxy at 2024-02-14 14:30
        self.long_cond1 = price_cond and cross_cond

        # 条件2：（归一化）斜率判断     
        # TODO XRH审计逻辑                                                                                           名字改为归一化斜率？ # remark by ZXY 改为 deltaZ ？  at 2024-02-14 10:40      #remark by XRH slope或k吧？2025-02-14 12:20      
        slope_threshold = Decimal(self.config.slope_entry) / self.sigma_t if self.sigma_t else Decimal('0')    # slope_threshold需要Decimal吗？ 为什么是str(slope_entry) # modified by ZXY  需要Decimal 保持数据一致性 ,str已删除 at 2024-02-14 10:47 
        self.long_cond2 = self.delta_z > slope_threshold                                                                                                         # 其他pass  02-13 19：20
        
        # 条件3：RSI过滤       # XRH pass 2025-02-14 12:20
        # TODO XRH审计逻辑
        self.long_cond3 = self.rsi < self.config.rsi_entry                                                         # XRH rsi_upper和lower本质是一个，可用config.rsi_entey代替  2025-02-13 19:40  # modified by ZXY  at 2024-02-14 10:50 
                                                                                                                 # XRH 其他pass  02-13 19：23
        
        # 记录每个条件的状态
        self.logger().info(
            f"做多条件检查 - "
            f"EMA位置和金叉: {'[满足]' if self.long_cond1 else '[不满足]'} "                                          # XRH 满足不满足直接1 0吧 好看些，下同 2025-02-14 12:20 # 我感觉注释多了一堆1 0可能不好看
            f"(位置: {'[满足]' if price_cond else '[不满足]'}, "
            f"金叉: {'[满足]' if cross_cond else '[不满足]'}), "
            
            f"斜率阈值: {'[满足]' if self.long_cond2 else '[不满足]'} "                                                         #同理，可否为归一化斜率或者斜率？   其他pass  02-13 19：20  # modified by ZXY  at 2024-02-14 10:50 
            f"(ΔZ={self.delta_z:.2f}, 阈值={self.config.slope_entry/self.sigma_t:.2f}), "          # XRH .2f .1f 啥意思 2025-02-14 12:20  # .2f表示保留两位小数的浮点数格式化，如 1.23。.1f表示保留一位小数，如 1.2 by ZXY at 2025-02-14 14:30
            f"RSI过滤: {'[满足]' if self.long_cond3 else '[不满足]'} "
            f"(RSI={self.rsi:.1f}, 阈值={self.config.rsi_entry})"                                            #RSI_upper-->entry   # modified by ZXY  at 2024-02-14 10:50 
        )
        
        # 记录最终决策
        self.long_conditions = self.long_cond1 and self.long_cond2 and self.long_cond3
        if self.long_conditions:
            self.logger().info("所有条件满足，触发做多信号")
        else:
            self.logger().info("未触发做多信号，等待下一次机会")
            
        return self.long_conditions

    def _check_entry_short_conditions(self) -> bool:
        # 记录当前市场状态
        self.logger().info(
            f"检查做空条件 - 当前价格: {self.current_price}, "
            f"BB中轨: {self.bb_middle}, BB下轨: {self.bb_lower}, "                           # XRH 这里如果是实时的，那是不是得为.iloc[-1]? 下同  2025-02-14 12:20 # pass 同上
            f"EMA5: {self.ema5}, EMA10: {self.ema10}, "
            f"RSI: {self.rsi}"
        )
        
        # 1. 检查市场状态
        if not self._check_market_ready():
            self.logger().info("市场状态未就绪，不执行做空")
            return False
            
        # 2. 检查风险指标
        if not self._check_risk_limits():
            self.logger().info("风险指标超限，不执行做空")
            return False
        
        # 3. 检查技术指标
        # 条件1：EMA5和EMA10的位置判断 + 死叉判断
        # TODO XRH审计逻辑                      同做多   其他pass  02-13 19：29  # ZXY 可以点了 需要Decimal 保持数据一致性 by ZXY at 2024-02-14 10:40
        bb_threshold = (self.config.bb_middle * Decimal(str(self.bb_middle)) + self.config.bb_up * Decimal(str(self.bb_upper))) / Decimal('5')
        price_cond = (self.ema5 > bb_threshold) or (self.ema10 > bb_threshold)
        cross_cond = (self.prev_ema5 - self.prev_ema10 > 0) and (self.ema5 - self.ema10 < 0)                # XRH 计算减法要套括号吗？ 2025-02-14 12:20 # pass 同上
        self.short_cond1 = price_cond and cross_cond
        
        # 条件2：（归一化）斜率判断       # XRH pass 2025-02-14 12:20
        # TODO XRH审计逻辑                         
        slope_threshold = Decimal(str(self.config.slope_entry)) / self.sigma_t if self.sigma_t else Decimal('0')           #这里不是exit， 这里的阈值和做多阈值一样  slope_enrty才是对的    # modified by ZXY  at 2024-02-14 10:53
        self.short_cond2 = -self.delta_z > slope_threshold
        
        # 条件3：RSI过滤
        # TODO XRH审计逻辑
        self.short_cond3 = self.rsi > (100 - self.config.rsi_entry)                # XRH 修改公式(100-config.rsi.enrty) 2025-02-14 12:20                                             #rsi_upper和lower本质是一个，用config.rsi_entey代替  # modified by ZXY  at 2024-02-14 10:53
        
        # 记录每个条件的状态
        self.logger().info(
            f"做空条件检查 - "
            f"EMA位置和死叉: {'[满足]' if self.short_cond1 else '[不满足]'} "                                # XRH 同理， 改成1/0为满足/不满足 2025-02-14 12:20 # 同上
            f"(位置: {'[满足]' if price_cond else '[不满足]'}, "
            f"死叉: {'[满足]' if cross_cond else '[不满足]'}), "
            
            f"归一化斜率: {'[满足]' if self.short_cond2 else '[不满足]'} "                                           # modified by ZXY  at 2024-02-14 10:53
            f"(ΔZ={-self.delta_z:.2f}, 阈值={self.config.slope_entry/self.sigma_t:.2f}), "             # modified by ZXY  at 2024-02-14 10:53
            f"RSI过滤: {'[满足]' if self.short_cond3 else '[不满足]'} "
            f"(RSI={self.rsi:.1f}, 阈值={self.config.rsi_entry})"                              ##RSI_lower-->entry                 02-13 19：29    # modified by ZXY  at 2024-02-14 10:53
        )
        
        # 记录最终决策
        self.short_conditions = self.short_cond1 and self.short_cond2 and self.short_cond3
        if self.short_conditions:
            self.logger().info("所有条件满足，触发做空信号")
        else:
            self.logger().info("未触发做空信号，等待下一次机会")
            
        return self.short_conditions
    
    def _get_spread(self) -> Decimal:
        """
        计算买卖价差
        
        计算方法：
        1. 取配置的买单价差和卖单价差的平均值
        2. 确保价差不小于最小价差要求
        
        Returns:
            Decimal: 计算得到的价差百分比
        """
        return self.config.min_order_spread  # 最小价差要求（0.01%）
        # return max(
        #     self.config.min_order_spread,  # 最小价差要求（0.01%）
            # TODO ZXY审查逻辑
            # ZXY 目前阶段不用考虑这两个参数，这两个参数是为了根据市场实时计算价差 at 2024-02-14 12:05
            # (self.config.bid_price_offset + self.config.ask_price_offset) / Decimal("2")  # 买卖价差的平均值  
        # )
        
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
        connector = self.connectors[self.config.connector_name]
        # 获取基础资产余额（如BTC-USDT中的BTC）
        #TODO 查看验证get_available_balance  #ZXY已经验证，用法正确 at 2024-02-14 11:47
        base_balance = connector.get_available_balance(self.config.trading_pair.split('-')[0])
        # 计算总价值 = 余额 * 当前价格
        return base_balance * Decimal(str(self.current_price))
         
    def _get_max_position_value(self) -> Decimal:
        """
        获取最大允许的持仓价值
        
        计算方法：
        1. 获取计价货币的可用余额（如USDT）
        2. 将余额乘以最大持仓比例和杠杆倍数
        
        Returns:
            Decimal: 最大允许的持仓价值
        """
        # 获取交易所连接器
        connector = self.connectors[self.config.connector_name]
        
        # 获取计价货币余额（如BTC-USDT中的USDT）
        quote_balance = connector.get_available_balance(self.config.trading_pair.split('-')[1])
        
        # 计算最大持仓价值 = 余额 * 最大持仓比例 * 杠杆倍数
        max_position = quote_balance * self.config.max_position_size * Decimal(str(self.config.leverage))
        
        self.logger().info(f"计算最大持仓 - 可用余额: {quote_balance}, 最大持仓比例: {self.config.max_position_size}, "
                          f"杠杆倍数: {self.config.leverage}, 最大持仓价值: {max_position}")
        
        return max_position
        
    def _create_long_proposal(self) -> CreateExecutorAction:
        """
        创建做多订单建议
        
        创建一个包含以下内容的做多订单建议:
        1. 做多价格：当前价格减去买入价差
        2. 做多数量：配置的订单数量
        3. 三重障碍设置：止损、止盈、时间限制
        
        Returns:
            CreateExecutorAction: 做多订单执行动作
        """
        # 1. 获取当前市场价格
        # TODO 使用最新根的开盘价
        current_open_price = Decimal(str(self.current_price))
        
        # 2. 计算做多价格 = 下一根 K 线开盘价下跌买入价差，考虑滑点
        spread = self._get_spread()
        entry_price = current_open_price * (1 - spread)
        
        # 3. 创建仓位执行器配置
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),               # 当前时间戳
            connector_name=self.config.connector_name,                      # 交易所
            trading_pair=self.config.trading_pair,                    # 交易对
            side=TradeType.BUY,                                       # 买入方向
            entry_price=entry_price,                                  # 买入价格
            amount=self.config.order_amount,                          # 买入数量
            leverage=self.config.leverage,                            # 杠杆倍数
            triple_barrier_conf=self.config.triple_barrier_config     # 三重障碍配置
        )
        
        # 4. 创建执行动作
        action = CreateExecutorAction(
            executor_config=executor_config
        )
        
        # 5. 记录日志
        self.logger().info(
            f"创建做多订单 - 交易对: {self.config.trading_pair}, "
            f"价格: {entry_price}, 数量: {self.config.order_amount}, "
            f"止损: {self.config.stop_loss}, 止盈: {self.config.take_profit}, "
            f"时间限制: {self.config.time_limit}秒"
        )
        
        return action
        
    def _create_short_proposal(self) -> CreateExecutorAction:
        """
        创建做空订单建议
        
        创建一个包含以下内容的做空订单建议:
        1. 做空价格：当前价格加上卖出价差
        2. 做空数量：配置的订单数量
        3. 三重障碍设置：止损、止盈、时间限制
        
        Returns:
            CreateExecutorAction: 做空订单执行动作
        """
        # 获取当前市场价格
        # TODO 使用最新根的开盘价
        current_open_price = Decimal(str(self.current_price))
        
        # 计算做空卖出价格 = 下一根 K 线开盘价上涨卖出价差，考虑滑点
        spread = self._get_spread()
        entry_price = current_open_price * (1 + spread)
        
        # 创建仓位执行器配置
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),               # 当前时间戳
            connector_name=self.config.connector_name,                      # 交易所
            trading_pair=self.config.trading_pair,                    # 交易对
            side=TradeType.SELL,                                      # 卖出方向
            entry_price=entry_price,                                  # 卖出价格
            amount=self.config.order_amount,                          # 卖出数量
            leverage=self.config.leverage,                            # 杠杆倍数
            triple_barrier_conf=self.config.triple_barrier_config     # 三重障碍配置
        )
        
        # 创建执行动作
        action = CreateExecutorAction(
            executor_config=executor_config
        )
        
        self.logger().info(
            f"创建做空订单 - 交易对: {self.config.trading_pair}, "
            f"价格: {entry_price}, 数量: {self.config.order_amount}, "
            f"止损: {self.config.stop_loss}, 止盈: {self.config.take_profit}, "
            f"时间限制: {self.config.time_limit}秒"
        )
        
        return action
        
    def _create_close_action(self, executor_id: str, position_side: PositionSide) -> CreateExecutorAction:
        """
        创建限价平仓动作
        
        Args:
            executor_id (str): 要停止的执行器ID
            position_side (PositionSide): 平仓方向，Long或Short
        
        Returns:
            CreateExecutorAction: 限价平仓动作
        """
        # 获取当前市场价格
        current_open_price = Decimal(str(self.current_price))
        
        # 计算平仓价格，考虑滑点
        spread = self._get_spread()
        if position_side == PositionSide.LONG:
            # 平多仓，在当前价格基础上减去价差，确保能成交
            close_price = current_open_price * (1 - spread)
        else:
            # 平空仓，在当前价格基础上加上价差，确保能成交
            close_price = current_open_price * (1 + spread)
        
        # 创建一个特殊的三重障碍配置用于平仓
        close_barrier_config = TripleBarrierConfig(
            stop_loss=Decimal("0.1"),                   # 设置一个较大的止损以防止触发
            take_profit=Decimal("0.1"),                 # 设置一个较大的止盈以防止触发
            time_limit=self.config.time_limit,          # 保持原有的时间限制
            open_order_type=OrderType.LIMIT,            # 开仓使用限价单
            take_profit_order_type=OrderType.LIMIT,     # 止盈使用限价单
            stop_loss_order_type=OrderType.LIMIT,       # 止损使用限价单
            time_limit_order_type=OrderType.LIMIT       # 时间限制平仓使用限价单
        )
        
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),               # 当前时间戳
            connector_name=self.config.connector_name,                      # 交易所
            trading_pair=self.config.trading_pair,                    # 交易对
            side=position_side,                                       # 平仓方向与持仓方向相反
            entry_price=close_price,                                  # 平仓价格
            amount=self.config.order_amount,                          # 平仓数量
            leverage=self.config.leverage,                            # 杠杆倍数
            triple_barrier_conf=close_barrier_config                  # 平仓的三重障碍配置
        )
        
        
        # 创建执行器动作
        return CreateExecutorAction(
            executor_id=executor_id,
            executor_config=executor_config
        )
    
    def _get_ema_data(self) -> Optional[tuple]:
        """
        获取EMA指标数据
        
        Returns:
            Optional[tuple]: (current_ema5, prev_ema5, current_ema10, prev_ema10)
            如果数据不完整返回None
        """
        current_ema5 = self.ema5[-1] if self.ema5 is not None else None
        prev_ema5 = self.ema5[-2] if self.ema5 is not None and len(self.ema5) > 1 else None
        current_ema10 = self.ema10[-1] if self.ema10 is not None else None
        prev_ema10 = self.ema10[-2] if self.ema10 is not None and len(self.ema10) > 1 else None
        
        if None in [current_ema5, prev_ema5, current_ema10, prev_ema10]:
            return None
        
        return current_ema5, prev_ema5, current_ema10, prev_ema10
    
    def _check_exit_long_conditions(self) -> bool:
        """
        检查平多条件
        
        Args:
            delta_z_current (float): EMA5的当前斜率
            ema_data (tuple): (current_ema5, prev_ema5, current_ema10, prev_ema10)
        
        Returns:
            bool: 是否触发平多信号
        """
        
        # 条件1：EMA5大于BB中轨
        cond_exit1_long = self.ema5 > self.bb_middle                  # XRH 注意我们需要的是条件蜡烛的判断，而不是最新蜡烛的判断，下所有条件都同 2025-02-14 12:20  # self.bb_middle已经是条件蜡烛的数据，不需要再用[-1] by ZXY at 2025-02-14 14:30
        
        # 条件2A：斜率小于阈值且RSI大于退出阈值
        '''
        1. 删除了abs
        2. slope_exit 15 -> 6
        3. 修改分母为sigma_t
        4. 修改rsi_upper为rsi_exit
         # modified by ZXY  at 2024-02-14 11:15
        '''
        cond_exit2a_long = (self.delta_z < (self.config.slope_exit / self.sigma_t) and       # XRH 1、千万不要abs！    2、这里弄混了，平多的slope_exit有问题  3、分母有问题是sigma_t 2025-02-13 19：40
                           self.rsi > self.config.rsi_exit)                                                # 不是rsi_upper,RSIexit        self.rsi > self.config.rsi_exit)  
        
        # 条件2B：EMA5和EMA10的交叉（由上穿变为下穿）
        cond_exit2b_long = ((self.prev_ema5 - self.prev_ema10 > 0) and 
                            (self.ema5 - self.ema10 < 0))
        
        # 综合条件   
        cond_exit2_long = cond_exit2a_long or cond_exit2b_long
        self.exit_long = cond_exit1_long and cond_exit2_long
        return self.exit_long
    
    def _check_exit_short_conditions(self) -> bool:
        """
        检查平空条件
        
        Args:
            delta_z_current (float): EMA5的当前斜率
            ema_data (tuple): (current_ema5, prev_ema5, current_ema10, prev_ema10)
        
        Returns:
            bool: 是否触发平空信号
        """

        # 条件1：EMA5小于BB中轨
        cond_exit1_short = self.ema5 < self.bb_middle                      # XRH 注意我们需要的是条件蜡烛的判断，而不是最新蜡烛的判断，下所有条件都同 2025-02-14 12:20  # self.bb_middle已经是条件蜡烛的数据，不需要再用[-1] by ZXY at 2025-02-14 14:30
        
        # 条件2A：斜率小于阈值且RSI小于退出阈值
        '''
        1. 删除了abs
        2. slope_exit 15 -> 6
        3. 修改分母为sigma_t
        4. 修改rsi_lower为rsi_exit
         # modified by ZXY  at 2024-02-14 11:15
        '''
        cond_exit2a_short = (-(self.delta_z) < (self.config.slope_exit / self.sigma_t) and    # XRH 1、千万不要abs！这是是负值的  2、这里弄混了，平多的slope_exit有问题  3、分母有问题sigma_t 2025-02-13 19:40
                            self.rsi < (100 - self.config.rsi_exit))               # XRH 修改100- self.config.rsi_exit 2025-02-14 12:20     #  XRH 不是rsi_lower   RSI_exit          self.rsi < (100-(self.config.rsi_lower)) 2025-02-13 19:40
        
        # 条件2B：EMA5和EMA10的交叉（由下穿变为上穿）
        cond_exit2b_short = ((self.prev_ema5 - self.prev_ema10 < 0) and 
                            (self.ema5 - self.ema10 > 0))
        
        # 综合条件
        cond_exit2_short = cond_exit2a_short or cond_exit2b_short
        self.exit_short = cond_exit1_short and cond_exit2_short
        return self.exit_short
    
    def _exit_actions_proposal(self) -> List[StopExecutorAction]:
        """
        创建停止交易动作建议
        
        在以下情况下会触发停止动作：
        1. 市场状态异常（流动性不足、波动率过高）
        2. 风险指标超限
        3. 技术指标平仓信号
        4. 紧急情况处理
        
        Returns:
            List[StopExecutorAction]: 停止动作列表，包含需要停止的执行器ID
        """
        # 检查风险限制
        if not self._check_risk_limits():
            self.logger().info("风险指标超限，准备停止交易")
            actions = []
            for controller_id, executors in self.executor_orchestrator.active_executors.items():
                for executor in executors:
                    actions.append(self._create_stop_action(executor.id))
            return actions
        
        # 获取当前持仓信息
        position = self._calculate_position_value()
        
        # 如果没有持仓，直接返回
        if position == Decimal("0"):
            return []
        
        # 检查平仓条件
        actions = []
        for controller_id, executors in self.executor_orchestrator.active_executors.items():
            for executor in executors:
                if position > Decimal("0") and self._check_exit_long_conditions():
                    self.logger().info(f"检测到平多信号，准备以限价单平仓 - 执行器ID: {executor.config.id or executor.id}")
                    actions.append(self._create_close_action(executor.config.id or executor.id, PositionSide.LONG))  # True for long position
                elif position < Decimal("0") and self._check_exit_short_conditions():
                    self.logger().info(f"检测到平空信号，准备以限价单平仓 - 执行器ID: {executor.config.id or executor.id}")
                    actions.append(self._create_close_action(executor.config.id or executor.id, PositionSide.SHORT))  # False for short position
        
        return actions
    
    def format_status(self) -> str:
        """
        返回策略状态的格式化字符串
        包含：市场信息、技术指标、仓位状态、性能指标等
        """
        if not self.ready:
            return "策略未就绪，等待数据初始化..."
            
        lines = []
        
        # 1. 基本信息
        lines.extend([
            "\n═══════════════ 布林带MA策略状态 ═══════════════",
            f"交易对: {self.config.connector_name}:{self.config.trading_pair}",
            f"策略状态: {'就绪' if self.ready else '未就绪'}",
            "───────────────────────────────────────────────"
        ])
        
        # 2.1 计算指标状态
        if all(v is not None for v in [self.delta_z, self.mu_t, self.sigma_t, self.ema5, self.ema10, 
                                       self.long_conditions, self.short_conditions, 
                                       self.long_cond1, self.long_cond2, self.long_cond3, 
                                       self.short_cond1, self.short_cond2, self.short_cond3]):
            lines.extend([
                "计算指标状态:",
                f"normslope:        {self.delta_z:.4f}",
                f"mu:               {self.mu_t:.4f}",
                f"sigma:            {self.sigma_t:.4f}",
                    
                f"EMA5:             {self.ema5:.4f}",
                f"EMA10:            {self.ema10:.4f}",
                
                f"entryLong:        {self.long_conditions}",
                f"entryCond1Long:   {self.long_cond1}",
                f"entryCond2Long:   {self.long_cond2}",
                f"entryCond3Long:   {self.long_cond3}",
                
                f"entryShort:       {self.short_conditions}",
                f"entryCond1Short:  {self.short_cond1}",
                f"entryCond2Short:  {self.short_cond2}",
                f"entryCond3Short:  {self.short_cond3}",
                
                # f"exitLong:         {self.exit_long}",
                # f"exitShort:        {self.exit_short}",
                "───────────────────────────────────────────────"
            ])
        
        # # 2.2 技术指标状态
        # if all(v is not None for v in [self.bb_upper, self.bb_middle, self.bb_lower, self.ema5, self.ema10, self.rsi]):
        #     lines.extend([
        #         "技术指标状态:",
        #         f"布林带上轨: {self.bb_upper:.4f}",
        #         f"布林带中轨: {self.bb_middle:.4f}",
        #         f"布林带下轨: {self.bb_lower:.4f}",
        #         f"EMA5: {self.ema5:.4f}",
        #         f"EMA10: {self.ema10:.4f}",
        #         f"RSI: {self.rsi:.2f}",
        #         "───────────────────────────────────────────────"
        #     ])
        
        # 3. 市场数据
        # if all(v is not None for v in [self.current_price, self.current_volume, self.current_volatility]):
        #     lines.extend([
        #         "市场数据:",
        #         f"当前价格: {self.current_price:.4f}",
        #         f"成交量: {self.current_volume:.2f}",
        #         f"波动率: {self.current_volatility*100:.2f}%",
        #         "───────────────────────────────────────────────"
        #     ])
        
        # 4. 仓位信息
        # try:
        #     connector = self.connectors[self.config.connector_name]
        #     base_asset, quote_asset = self.config.trading_pair.split('-')
        #     base_balance = connector.get_available_balance(base_asset)
        #     quote_balance = connector.get_available_balance(quote_asset)
        #     position_value = self._calculate_position_value()
        #     max_position = self._get_max_position_value()
            
        #     lines.extend([
        #         "仓位信息:",
        #         f"基础货币({base_asset})余额: {base_balance:.4f}",
        #         f"计价货币({quote_asset})余额: {quote_balance:.4f}",
        #         f"当前持仓价值: {position_value:.4f} {quote_asset}",
        #         f"最大持仓限制: {max_position:.4f} {quote_asset}",
        #         f"仓位使用率: {(position_value/max_position*100 if max_position else 0):.2f}%",
        #         "───────────────────────────────────────────────"
        #     ])
        # except Exception as e:
        #     lines.extend([
        #         "仓位信息获取失败:",
        #         f"错误: {str(e)}",
        #         "───────────────────────────────────────────────"
        #     ])
        
        # 5. 性能指标
        # if hasattr(self, 'metrics'):
        #     lines.extend([
        #         "策略性能:",
        #         f"盈亏: {self.metrics.get('pnl', 0):.4f} {quote_asset}",
        #         f"胜率: {(self.metrics.get('win_count', 0) / (self.metrics.get('win_count', 0) + self.metrics.get('loss_count', 0)) * 100 if (self.metrics.get('win_count', 0) + self.metrics.get('loss_count', 0)) > 0 else 0):.2f}%",
        #         "───────────────────────────────────────────────"             # XRH 这里怎么获取的？ 2025-02-14 12:20 #此处后续优化 ZXY at 2024-02-14 14:30
        #     ])
        
        # 6. 风险指标
        # lines.extend([
        #     "风险控制:",
        #     f"杠杆倍数: {self.config.leverage}x",
        #     f"止损设置: {self.config.stop_loss*100:.2f}%",                    #  XRH 为什么是*100  2025-02-14 12:20 # 将小数转换为百分比，如 0.05 -> 5% by ZXY at 2025-02-14 14:30
        #     f"止盈设置: {self.config.take_profit*100:.2f}%",
        #     f"持仓时间限制: {self.config.time_limit/3600:.1f}小时",            #  XRH 为什么是/3600  2025-02-14 12:20 # 将秒转换为小时，3600秒=1小时 by ZXY at 2025-02-14 14:30
        #     "═══════════════════════════════════════════════"
        # ])
        
        return "\n".join(lines)