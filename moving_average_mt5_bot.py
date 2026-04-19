"""
Moving Average Trading Bot
Supports: Single MA, Dual MA Crossover, Triple MA strategies
+ MT5 live trading integration
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


# ─────────────────────────────────────────────
#  ENUMS & DATA CLASSES
# ─────────────────────────────────────────────


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MAType(Enum):
    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"


class Strategy(Enum):
    SINGLE = "SINGLE"  # Price vs 1 MA
    DUAL = "DUAL"  # 2 MAs crossover
    TRIPLE = "TRIPLE"  # 3 MAs alignment


@dataclass
class BotConfig:
    strategy: Strategy = Strategy.DUAL
    ma_type: MAType = MAType.EMA
    fast_period: int = 9
    slow_period: int = 21
    trend_period: int = 50  # Only used in TRIPLE strategy
    risk_pct: float = 0.01  # 1% of balance per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    reward_ratio: float = 2.0  # Risk:Reward (e.g. 2 = 2x the risk)


@dataclass
class MT5Config:
    symbol: str = "EURUSD"
    timeframe: int = mt5.TIMEFRAME_M15 if mt5 else 0
    bars: int = 300
    deviation: int = 20
    magic: int = 20260419
    poll_seconds: int = 30


@dataclass
class Position:
    side: str  # "long" or "short"
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    is_open: bool = True


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit"


@dataclass
class Portfolio:
    balance: float
    initial: float
    position: Optional[Position] = None
    trades: list = field(default_factory=list)

    @property
    def total_return_pct(self):
        return ((self.balance - self.initial) / self.initial) * 100

    @property
    def win_rate(self):
        if not self.trades:
            return 0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return (wins / len(self.trades)) * 100

    @property
    def max_drawdown(self):
        if not self.trades:
            return 0
        peak = self.initial
        max_dd = 0
        running = self.initial
        for t in self.trades:
            running += t.pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd


# ─────────────────────────────────────────────
#  INDICATOR ENGINE
# ─────────────────────────────────────────────


class IndicatorEngine:
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(prices: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return prices.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def compute(self, prices: pd.Series, period: int, ma_type: MAType) -> pd.Series:
        if ma_type == MAType.SMA:
            return self.sma(prices, period)
        if ma_type == MAType.EMA:
            return self.ema(prices, period)
        if ma_type == MAType.WMA:
            return self.wma(prices, period)
        raise ValueError(f"Unknown MA type: {ma_type}")


# ─────────────────────────────────────────────
#  SIGNAL GENERATOR
# ─────────────────────────────────────────────


class SignalGenerator:
    def __init__(self, config: BotConfig):
        self.config = config
        self.engine = IndicatorEngine()

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a DataFrame with at least a 'close' column.
        Returns the DataFrame with MA columns and a 'signal' column added.
        """
        prices = df["close"]
        c = self.config

        if c.strategy == Strategy.SINGLE:
            df["ma"] = self.engine.compute(prices, c.slow_period, c.ma_type)
            df["signal"] = df.apply(lambda row: self._single_signal(row, df), axis=1)

        elif c.strategy == Strategy.DUAL:
            df["fast_ma"] = self.engine.compute(prices, c.fast_period, c.ma_type)
            df["slow_ma"] = self.engine.compute(prices, c.slow_period, c.ma_type)
            df["signal"] = self._dual_signals(df)

        elif c.strategy == Strategy.TRIPLE:
            df["fast_ma"] = self.engine.compute(prices, c.fast_period, c.ma_type)
            df["slow_ma"] = self.engine.compute(prices, c.slow_period, c.ma_type)
            df["trend_ma"] = self.engine.compute(prices, c.trend_period, c.ma_type)
            df["signal"] = self._triple_signals(df)

        return df

    # ── Single MA: price crosses above/below MA ──
    def _single_signal(self, row, df: pd.DataFrame) -> str:
        idx = df.index.get_loc(row.name)
        if idx < 1:
            return Signal.HOLD.value
        prev = df.iloc[idx - 1]
        if prev["close"] < prev["ma"] and row["close"] > row["ma"]:
            return Signal.BUY.value
        if prev["close"] > prev["ma"] and row["close"] < row["ma"]:
            return Signal.SELL.value
        return Signal.HOLD.value

    # ── Dual MA: fast crosses slow ──
    def _dual_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["fast_ma"]
        slow = df["slow_ma"]
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)

        buy = (prev_fast < prev_slow) & (fast > slow)  # Golden Cross
        sell = (prev_fast > prev_slow) & (fast < slow)  # Death Cross

        return np.where(
            buy, Signal.BUY.value, np.where(sell, Signal.SELL.value, Signal.HOLD.value)
        )

    # ── Triple MA: all 3 aligned ──
    def _triple_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["fast_ma"]
        slow = df["slow_ma"]
        trend = df["trend_ma"]

        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)

        buy = (prev_fast <= prev_slow) & (fast > slow) & (fast > trend)
        sell = (prev_fast >= prev_slow) & (fast < slow) & (fast < trend)

        return np.where(
            buy, Signal.BUY.value, np.where(sell, Signal.SELL.value, Signal.HOLD.value)
        )


# ─────────────────────────────────────────────
#  RISK MANAGER
# ─────────────────────────────────────────────


class RiskManager:
    def __init__(self, config: BotConfig):
        self.config = config

    def position_size(self, balance: float, entry_price: float) -> float:
        """How many units to buy based on risk %"""
        risk_amount = balance * self.config.risk_pct
        risk_per_unit = entry_price * self.config.stop_loss_pct
        return risk_amount / risk_per_unit

    def stop_loss(self, entry_price: float, side: str) -> float:
        sl = self.config.stop_loss_pct
        return entry_price * (1 - sl) if side == "long" else entry_price * (1 + sl)

    def take_profit(self, entry_price: float, side: str) -> float:
        tp = self.config.stop_loss_pct * self.config.reward_ratio
        return entry_price * (1 + tp) if side == "long" else entry_price * (1 - tp)


# ─────────────────────────────────────────────
#  TRADING ENGINE (Backtest)
# ─────────────────────────────────────────────


class TradingEngine:
    def __init__(self, config: BotConfig, initial_balance: float = 10_000):
        self.config = config
        self.signals = SignalGenerator(config)
        self.risk = RiskManager(config)
        self.portfolio = Portfolio(balance=initial_balance, initial=initial_balance)

    def run(self, df: pd.DataFrame) -> Portfolio:
        """Run backtest over historical OHLCV data."""
        df = self.signals.compute_signals(df.copy())

        for _, row in df.iterrows():
            if pd.isna(row.get("fast_ma", row.get("ma", None))):
                continue  # Skip rows where MA not yet computed

            price = row["close"]
            signal = row["signal"]
            pos = self.portfolio.position

            # ── Check stops on open position ──
            if pos and pos.is_open:
                if pos.side == "long":
                    if price <= pos.stop_loss:
                        self._close_position(price, "stop_loss")
                        continue
                    if price >= pos.take_profit:
                        self._close_position(price, "take_profit")
                        continue
                elif pos.side == "short":
                    if price >= pos.stop_loss:
                        self._close_position(price, "stop_loss")
                        continue
                    if price <= pos.take_profit:
                        self._close_position(price, "take_profit")
                        continue

            # ── Act on new signal ──
            if signal == Signal.BUY.value:
                if pos and pos.is_open and pos.side == "short":
                    self._close_position(price, "signal")
                if not self.portfolio.position:
                    self._open_position("long", price)

            elif signal == Signal.SELL.value:
                if pos and pos.is_open and pos.side == "long":
                    self._close_position(price, "signal")
                if not self.portfolio.position:
                    self._open_position("short", price)

        # Close any remaining position at end of data
        if self.portfolio.position:
            self._close_position(df.iloc[-1]["close"], "end_of_data")

        return self.portfolio

    def _open_position(self, side: str, price: float):
        size = self.risk.position_size(self.portfolio.balance, price)
        self.portfolio.position = Position(
            side=side,
            entry_price=price,
            size=size,
            stop_loss=self.risk.stop_loss(price, side),
            take_profit=self.risk.take_profit(price, side),
        )

    def _close_position(self, price: float, reason: str):
        pos = self.portfolio.position
        if not pos:
            return

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - price) * pos.size

        pnl_pct = (pnl / (pos.entry_price * pos.size)) * 100
        self.portfolio.balance += pnl
        self.portfolio.trades.append(
            Trade(
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
            )
        )
        self.portfolio.position = None


# ─────────────────────────────────────────────
#  MT5 LIVE TRADING CONNECTOR
# ─────────────────────────────────────────────


class MT5Trader:
    def __init__(self, bot_config: BotConfig, mt5_config: MT5Config):
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package is not installed. Run: pip install MetaTrader5"
            )
        self.bot_config = bot_config
        self.mt5_config = mt5_config
        self.signals = SignalGenerator(bot_config)

    def connect(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        if not mt5.initialize(login=login, password=password, server=server):
            code, msg = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {code} {msg}")

        if not mt5.symbol_select(self.mt5_config.symbol, True):
            code, msg = mt5.last_error()
            raise RuntimeError(f"Cannot select symbol {self.mt5_config.symbol}: {code} {msg}")

    def shutdown(self):
        mt5.shutdown()

    def fetch_ohlcv(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(
            self.mt5_config.symbol,
            self.mt5_config.timeframe,
            0,
            self.mt5_config.bars,
        )
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise RuntimeError(f"copy_rates_from_pos failed: {code} {msg}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["time", "open", "high", "low", "close", "volume"]].set_index("time")

    def latest_signal(self) -> str:
        df = self.fetch_ohlcv()
        out = self.signals.compute_signals(df.copy())
        return out.iloc[-1]["signal"]

    def has_open_position(self) -> Optional[int]:
        positions = mt5.positions_get(symbol=self.mt5_config.symbol)
        if not positions:
            return None
        for p in positions:
            if p.magic == self.mt5_config.magic:
                return p.type
        return None

    def calc_volume(self, price: float) -> float:
        acc = mt5.account_info()
        if acc is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"account_info failed: {code} {msg}")
        risk = RiskManager(self.bot_config)
        units = risk.position_size(acc.balance, price)

        info = mt5.symbol_info(self.mt5_config.symbol)
        if info is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"symbol_info failed: {code} {msg}")

        # Approximate unit->lot conversion using contract size.
        lots = max(info.volume_min, units / info.trade_contract_size)
        lots = min(lots, info.volume_max)
        step = info.volume_step
        return round(lots / step) * step

    def send_order(self, side: str):
        tick = mt5.symbol_info_tick(self.mt5_config.symbol)
        if tick is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"symbol_info_tick failed: {code} {msg}")

        price = tick.ask if side == "buy" else tick.bid
        rm = RiskManager(self.bot_config)
        volume = self.calc_volume(price)
        if volume <= 0:
            raise RuntimeError("Calculated volume is <= 0")

        sl = rm.stop_loss(price, "long" if side == "buy" else "short")
        tp = rm.take_profit(price, "long" if side == "buy" else "short")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.mt5_config.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.mt5_config.deviation,
            "magic": self.mt5_config.magic,
            "comment": "ma-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"order_send failed: {result}")

    def close_by_signal(self, open_type: int, signal: str):
        if signal == Signal.BUY.value and open_type == mt5.POSITION_TYPE_SELL:
            self.send_order("buy")
        elif signal == Signal.SELL.value and open_type == mt5.POSITION_TYPE_BUY:
            self.send_order("sell")

    def run_forever(self):
        print(f"Running MT5 MA bot on {self.mt5_config.symbol}...")
        while True:
            try:
                signal = self.latest_signal()
                open_type = self.has_open_position()
                print(f"Signal={signal}, open_position={open_type}")

                if signal == Signal.BUY.value and open_type is None:
                    self.send_order("buy")
                    print("BUY order sent")
                elif signal == Signal.SELL.value and open_type is None:
                    self.send_order("sell")
                    print("SELL order sent")
                elif open_type is not None:
                    self.close_by_signal(open_type, signal)

            except Exception as exc:
                print(f"Loop error: {exc}")

            time.sleep(self.mt5_config.poll_seconds)


# ─────────────────────────────────────────────
#  RESULTS PRINTER
# ─────────────────────────────────────────────


def print_results(portfolio: Portfolio):
    trades = portfolio.trades
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    pf = (
        abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
        if losses
        else float("inf")
    )

    print("=" * 45)
    print("          BACKTEST RESULTS")
    print("=" * 45)
    print(f"  Starting Balance : ${portfolio.initial:>10,.2f}")
    print(f"  Final Balance    : ${portfolio.balance:>10,.2f}")
    print(f"  Total Return     : {portfolio.total_return_pct:>10.2f}%")
    print(f"  Max Drawdown     : {portfolio.max_drawdown:>10.2f}%")
    print("-" * 45)
    print(f"  Total Trades     : {len(trades):>10}")
    print(f"  Win Rate         : {portfolio.win_rate:>10.1f}%")
    print(f"  Profit Factor    : {pf:>10.2f}")
    print(f"  Avg Win          : ${avg_win:>10,.2f}")
    print(f"  Avg Loss         : ${avg_loss:>10,.2f}")
    print("=" * 45)


if __name__ == "__main__":
    # BACKTEST EXAMPLE
    np.random.seed(42)
    n = 500
    closes = 30000 + np.cumsum(np.random.randn(n) * 200)
    df = pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": np.random.randint(100, 1000, n),
        }
    )

    config = BotConfig(
        strategy=Strategy.DUAL,
        ma_type=MAType.EMA,
        fast_period=9,
        slow_period=21,
        risk_pct=0.01,
        stop_loss_pct=0.02,
        reward_ratio=2.0,
    )

    engine = TradingEngine(config, initial_balance=10_000)
    portfolio = engine.run(df)
    print_results(portfolio)

    # LIVE MT5 EXAMPLE (uncomment to use)
    # mt5_cfg = MT5Config(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, bars=300)
    # trader = MT5Trader(config, mt5_cfg)
    # trader.connect(login=12345678, password="your_password", server="YourBroker-Server")
    # try:
    #     trader.run_forever()
    # finally:
    #     trader.shutdown()
