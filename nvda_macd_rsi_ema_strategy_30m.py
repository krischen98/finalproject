"""
NVDA MACD + RSI + EMA Combined Trading Strategy — 30-Minute Bars
=================================================================
Fetches NVDA 30m intraday data, computes MACD, RSI, and EMA indicators,
backtests multiple parameter variants tuned for intraday trading,
and reports the best one.

Note: Yahoo Finance provides ~60 days of 30-min data.
Indicator periods and risk parameters are scaled for intraday moves.
"""

import json
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_nvda_30m(period: str = "max", interval: str = "30m") -> pd.DataFrame:
    """Download NVDA 30-minute OHLCV data from Yahoo Finance."""
    ticker = yf.Ticker("NVDA")
    df = ticker.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"Fetched {len(df)} 30-min bars for NVDA  "
          f"({df.index[0]} -> {df.index[-1]})")
    n_days = (df.index[-1] - df.index[0]).days
    print(f"  Spanning ~{n_days} calendar days  ({len(df)} bars, "
          f"~{len(df)//13} trading days)")
    return df

# ---------------------------------------------------------------------------
# 2. TECHNICAL INDICATORS
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series, fast: int, slow: int,
                 signal: int) -> pd.DataFrame:
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": histogram,
    }, index=close.index)


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame, macd_fast: int, macd_slow: int,
                   macd_signal: int, rsi_period: int,
                   ema_short: int, ema_long: int) -> pd.DataFrame:
    """Attach all indicators to the dataframe."""
    df = df.copy()
    macd_df = compute_macd(df["Close"], macd_fast, macd_slow, macd_signal)
    df["MACD"] = macd_df["MACD"]
    df["MACD_Signal"] = macd_df["MACD_Signal"]
    df["MACD_Hist"] = macd_df["MACD_Hist"]
    df["RSI"] = compute_rsi(df["Close"], rsi_period)
    df["EMA_Short"] = compute_ema(df["Close"], ema_short)
    df["EMA_Long"] = compute_ema(df["Close"], ema_long)
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------------------------
# 3. STRATEGY SIGNAL GENERATION
# ---------------------------------------------------------------------------

@dataclass
class StrategyParams:
    """All tuneable parameters for the intraday MACD+RSI+EMA strategy."""
    # MACD (shorter periods for 30m bars)
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 5
    # RSI
    rsi_period: int = 7
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    # EMA crossover
    ema_short: int = 5
    ema_long: int = 13
    # Risk management (tighter for intraday)
    stop_loss_pct: float = 0.015      # 1.5% trailing stop
    take_profit_pct: float = 0.03     # 3% take-profit
    # Label
    name: str = ""

    def short_name(self) -> str:
        if self.name:
            return self.name
        return (f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})"
                f"_RSI({self.rsi_period},{self.rsi_oversold:.0f},{self.rsi_overbought:.0f})"
                f"_EMA({self.ema_short},{self.ema_long})"
                f"_SL{self.stop_loss_pct:.1%}_TP{self.take_profit_pct:.1%}")


def generate_signals_strict(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """
    Strict entry logic — all three indicators must agree:
      BUY  when MACD crosses above signal AND RSI < oversold AND EMA bullish
      SELL when MACD crosses below signal AND RSI > overbought AND EMA bearish
    """
    df = df.copy()
    df["MACD_Cross_Up"] = (df["MACD"] > df["MACD_Signal"]) & \
                          (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    df["MACD_Cross_Down"] = (df["MACD"] < df["MACD_Signal"]) & \
                            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))
    df["RSI_Oversold"] = df["RSI"] < params.rsi_oversold
    df["RSI_Overbought"] = df["RSI"] > params.rsi_overbought
    df["EMA_Bullish"] = df["EMA_Short"] > df["EMA_Long"]
    df["EMA_Bearish"] = df["EMA_Short"] < df["EMA_Long"]

    df["Signal"] = 0
    df.loc[df["MACD_Cross_Up"] & df["RSI_Oversold"] & df["EMA_Bullish"], "Signal"] = 1
    df.loc[df["MACD_Cross_Down"] & df["RSI_Overbought"] & df["EMA_Bearish"], "Signal"] = -1
    return df


def generate_signals_relaxed(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """
    Relaxed entry logic — 2-of-3 confirmation:
      BUY  when at least 2 of: MACD cross up, RSI in oversold zone, EMA bullish
      SELL when at least 2 of the bearish equivalents
    """
    df = df.copy()
    df["MACD_Cross_Up"] = (df["MACD"] > df["MACD_Signal"]) & \
                          (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    df["MACD_Cross_Down"] = (df["MACD"] < df["MACD_Signal"]) & \
                            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

    # Wider RSI band for relaxed mode
    df["RSI_Oversold"] = df["RSI"] < params.rsi_oversold + 10
    df["RSI_Overbought"] = df["RSI"] > params.rsi_overbought - 10
    df["EMA_Bullish"] = df["EMA_Short"] > df["EMA_Long"]
    df["EMA_Bearish"] = df["EMA_Short"] < df["EMA_Long"]

    buy_score = (df["MACD_Cross_Up"].astype(int)
                 + df["RSI_Oversold"].astype(int)
                 + df["EMA_Bullish"].astype(int))
    sell_score = (df["MACD_Cross_Down"].astype(int)
                  + df["RSI_Overbought"].astype(int)
                  + df["EMA_Bearish"].astype(int))

    df["Signal"] = 0
    df.loc[buy_score >= 2, "Signal"] = 1
    df.loc[sell_score >= 2, "Signal"] = -1
    return df


def generate_signals_macd_rsi(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """
    MACD + RSI only (no EMA filter). Good for ranging intraday markets.
      BUY  when MACD crosses up AND RSI < oversold
      SELL when MACD crosses down AND RSI > overbought
    """
    df = df.copy()
    df["MACD_Cross_Up"] = (df["MACD"] > df["MACD_Signal"]) & \
                          (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    df["MACD_Cross_Down"] = (df["MACD"] < df["MACD_Signal"]) & \
                            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))
    df["RSI_Oversold"] = df["RSI"] < params.rsi_oversold
    df["RSI_Overbought"] = df["RSI"] > params.rsi_overbought

    df["Signal"] = 0
    df.loc[df["MACD_Cross_Up"] & df["RSI_Oversold"], "Signal"] = 1
    df.loc[df["MACD_Cross_Down"] & df["RSI_Overbought"], "Signal"] = -1
    return df

# ---------------------------------------------------------------------------
# 4. BACKTESTING ENGINE
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    params: StrategyParams
    signal_mode: str = ""
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    buy_hold_return_pct: float = 0.0
    avg_trade_return_pct: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: list = field(default_factory=list)


def backtest(df: pd.DataFrame, params: StrategyParams,
             initial_capital: float = 100_000.0,
             signal_mode: str = "relaxed") -> BacktestResult:
    """
    Event-driven backtest for intraday 30-min bars.
    Position sizing: 100% of equity per trade (long only).
    """
    df_ind = add_indicators(df, params.macd_fast, params.macd_slow,
                            params.macd_signal, params.rsi_period,
                            params.ema_short, params.ema_long)

    if signal_mode == "strict":
        df_sig = generate_signals_strict(df_ind, params)
    elif signal_mode == "macd_rsi":
        df_sig = generate_signals_macd_rsi(df_ind, params)
    else:
        df_sig = generate_signals_relaxed(df_ind, params)

    capital = initial_capital
    position = 0
    entry_price = 0.0
    high_water = 0.0
    equity = []
    trades = []

    for i in range(len(df_sig)):
        row = df_sig.iloc[i]
        price = row["Close"]
        signal = row["Signal"]
        dt = df_sig.index[i]

        if position > 0:
            current_val = position * price
            high_water = max(high_water, current_val)
            dd_from_peak = (high_water - current_val) / high_water
            ret_from_entry = (price - entry_price) / entry_price

            if dd_from_peak >= params.stop_loss_pct or \
               ret_from_entry >= params.take_profit_pct or \
               signal == -1:
                capital = position * price
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({
                    "exit_date": dt,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "reason": ("SL" if dd_from_peak >= params.stop_loss_pct
                               else "TP" if ret_from_entry >= params.take_profit_pct
                               else "Signal"),
                })
                position = 0
                entry_price = 0.0
                high_water = 0.0

        if position == 0 and signal == 1:
            position = capital / price
            entry_price = price
            high_water = capital
            capital = 0.0
            trades.append({"entry_date": dt, "entry_price": price})

        eq = capital + position * price
        equity.append(eq)

    # Close open position at last bar
    if position > 0:
        last_price = df_sig.iloc[-1]["Close"]
        capital = position * last_price
        pnl_pct = (last_price - entry_price) / entry_price * 100
        trades.append({
            "exit_date": df_sig.index[-1],
            "exit_price": last_price,
            "pnl_pct": pnl_pct,
            "reason": "EOD",
        })
        position = 0

    equity_series = pd.Series(equity, index=df_sig.index)

    final_equity = equity[-1] if equity else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100

    n_days = (df_sig.index[-1] - df_sig.index[0]).days
    n_years = max(n_days / 365.25, 0.01)
    annual_return = ((final_equity / initial_capital) ** (1 / n_years) - 1) * 100

    # Sharpe — annualise using √(bars_per_year).  ~13 bars/day × 252 days
    bar_returns = equity_series.pct_change().dropna()
    bars_per_year = 13 * 252  # 30-min bars
    sharpe = (bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year)
              if bar_returns.std() > 0 else 0.0)

    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_dd = drawdowns.min() * 100

    completed = [t for t in trades if "pnl_pct" in t]
    wins = [t for t in completed if t["pnl_pct"] > 0]
    losses = [t for t in completed if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(completed) * 100 if completed else 0
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_trade_ret = np.mean([t["pnl_pct"] for t in completed]) if completed else 0

    bh_return = (df_sig.iloc[-1]["Close"] / df_sig.iloc[0]["Close"] - 1) * 100

    return BacktestResult(
        params=params,
        signal_mode=signal_mode,
        total_return_pct=total_return,
        annual_return_pct=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate_pct=win_rate,
        total_trades=len(completed),
        profit_factor=profit_factor,
        buy_hold_return_pct=bh_return,
        avg_trade_return_pct=avg_trade_ret,
        equity_curve=equity_series,
        trades=trades,
    )

# ---------------------------------------------------------------------------
# 5. PARAMETER VARIANTS — tuned for 30-min intraday
# ---------------------------------------------------------------------------

def build_variants() -> list[tuple[StrategyParams, str]]:
    """
    Return a list of (StrategyParams, signal_mode) variants.
    Intraday 30m bars need:
      - Faster indicator periods (5-13 bars = 2.5h - 6.5h)
      - Tighter SL/TP (0.5% - 4% for intraday swings)
    """
    variants = []

    # ===================== RELAXED (2-of-3) =====================

    # V1: Intraday classic relaxed
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=35, rsi_overbought=65,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.015, take_profit_pct=0.03,
        name="V1_Intraday_Classic",
    ), "relaxed"))

    # V2: Scalper — very fast, tight risk
    variants.append((StrategyParams(
        macd_fast=3, macd_slow=8, macd_signal=3,
        rsi_period=5, rsi_oversold=30, rsi_overbought=70,
        ema_short=3, ema_long=8,
        stop_loss_pct=0.008, take_profit_pct=0.015,
        name="V2_Scalper",
    ), "relaxed"))

    # V3: Medium-term intraday
    variants.append((StrategyParams(
        macd_fast=8, macd_slow=17, macd_signal=6,
        rsi_period=10, rsi_oversold=35, rsi_overbought=65,
        ema_short=8, ema_long=21,
        stop_loss_pct=0.02, take_profit_pct=0.04,
        name="V3_MediumIntraday",
    ), "relaxed"))

    # V4: Wide TP — let winners run
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=35, rsi_overbought=65,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.015, take_profit_pct=0.06,
        name="V4_WideTP",
    ), "relaxed"))

    # V5: Tight SL — cut losses fast
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=35, rsi_overbought=65,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.008, take_profit_pct=0.03,
        name="V5_TightSL",
    ), "relaxed"))

    # V6: Wider RSI bands — fewer but higher-conviction entries
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=25, rsi_overbought=75,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.015, take_profit_pct=0.04,
        name="V6_WideRSI",
    ), "relaxed"))

    # V7: Momentum burst — ultra-fast indicators
    variants.append((StrategyParams(
        macd_fast=4, macd_slow=9, macd_signal=3,
        rsi_period=5, rsi_oversold=30, rsi_overbought=70,
        ema_short=4, ema_long=9,
        stop_loss_pct=0.01, take_profit_pct=0.025,
        name="V7_MomentumBurst",
    ), "relaxed"))

    # V8: Swing intraday — longer hold, wider risk
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=14, rsi_oversold=35, rsi_overbought=65,
        ema_short=10, ema_long=26,
        stop_loss_pct=0.025, take_profit_pct=0.06,
        name="V8_SwingIntraday",
    ), "relaxed"))

    # V9: Aggressive — generous TP, moderate SL
    variants.append((StrategyParams(
        macd_fast=5, macd_slow=12, macd_signal=4,
        rsi_period=6, rsi_oversold=30, rsi_overbought=70,
        ema_short=5, ema_long=12,
        stop_loss_pct=0.012, take_profit_pct=0.05,
        name="V9_Aggressive",
    ), "relaxed"))

    # V10: Balanced — classic MACD with intraday risk
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=9, rsi_oversold=30, rsi_overbought=70,
        ema_short=6, ema_long=15,
        stop_loss_pct=0.015, take_profit_pct=0.035,
        name="V10_Balanced",
    ), "relaxed"))

    # ===================== MACD + RSI only (no EMA filter) =====================

    # V11: MACD+RSI scalper
    variants.append((StrategyParams(
        macd_fast=4, macd_slow=9, macd_signal=3,
        rsi_period=5, rsi_oversold=35, rsi_overbought=65,
        ema_short=4, ema_long=9,
        stop_loss_pct=0.01, take_profit_pct=0.02,
        name="V11_MACD_RSI_Scalp",
    ), "macd_rsi"))

    # V12: MACD+RSI medium
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=30, rsi_overbought=70,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.015, take_profit_pct=0.035,
        name="V12_MACD_RSI_Med",
    ), "macd_rsi"))

    # V13: MACD+RSI wide TP
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=35, rsi_overbought=65,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.012, take_profit_pct=0.05,
        name="V13_MACD_RSI_WideTP",
    ), "macd_rsi"))

    # ===================== STRICT (3-of-3) =====================

    # V14: Strict with fast params (may actually fire on 30m)
    variants.append((StrategyParams(
        macd_fast=3, macd_slow=8, macd_signal=3,
        rsi_period=5, rsi_oversold=35, rsi_overbought=65,
        ema_short=3, ema_long=8,
        stop_loss_pct=0.01, take_profit_pct=0.025,
        name="V14_Strict_Fast",
    ), "strict"))

    # V15: Strict with ultra-wide RSI to allow more entries
    variants.append((StrategyParams(
        macd_fast=4, macd_slow=9, macd_signal=3,
        rsi_period=5, rsi_oversold=45, rsi_overbought=55,
        ema_short=4, ema_long=9,
        stop_loss_pct=0.012, take_profit_pct=0.03,
        name="V15_Strict_WideRSI",
    ), "strict"))

    # V16: Strict classic intraday
    variants.append((StrategyParams(
        macd_fast=6, macd_slow=13, macd_signal=5,
        rsi_period=7, rsi_oversold=40, rsi_overbought=60,
        ema_short=5, ema_long=13,
        stop_loss_pct=0.015, take_profit_pct=0.04,
        name="V16_Strict_Classic",
    ), "strict"))

    return variants

# ---------------------------------------------------------------------------
# 6. RANKING & REPORTING
# ---------------------------------------------------------------------------

def rank_results(results: list[BacktestResult]) -> list[BacktestResult]:
    """
    Composite ranking score:
      0.25*norm(total_return) + 0.25*norm(sharpe) +
      0.20*norm(-max_dd) + 0.15*norm(win_rate) + 0.15*norm(profit_factor)
    """
    # Filter out variants with 0 trades
    active = [r for r in results if r.total_trades > 0]
    inactive = [r for r in results if r.total_trades == 0]

    if not active:
        return results

    def safe_norm(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn == 0:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    ret = [r.total_return_pct for r in active]
    sha = [r.sharpe_ratio for r in active]
    dd = [-r.max_drawdown_pct for r in active]
    wr = [r.win_rate_pct for r in active]
    pf = [min(r.profit_factor, 10) for r in active]

    scores = (0.25 * safe_norm(ret) + 0.25 * safe_norm(sha)
              + 0.20 * safe_norm(dd) + 0.15 * safe_norm(wr)
              + 0.15 * safe_norm(pf))

    paired = sorted(zip(scores, active), key=lambda x: x[0], reverse=True)
    return [r for _, r in paired] + inactive


def print_results_table(results: list[BacktestResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 140)
    print(f"{'Variant':<28} {'Mode':<10} {'Return%':>9} {'Annual%':>9} {'Sharpe':>8} "
          f"{'MaxDD%':>9} {'WinRate%':>9} {'Trades':>7} {'PF':>7} "
          f"{'AvgTrd%':>9} {'B&H%':>9}")
    print("=" * 140)
    for r in results:
        pf_str = f"{r.profit_factor:>7.2f}" if r.profit_factor < 100 else "    inf"
        print(f"{r.params.short_name():<28} {r.signal_mode:<10} "
              f"{r.total_return_pct:>+9.2f} "
              f"{r.annual_return_pct:>+9.1f} {r.sharpe_ratio:>8.2f} "
              f"{r.max_drawdown_pct:>9.2f} {r.win_rate_pct:>9.1f} "
              f"{r.total_trades:>7} {pf_str} "
              f"{r.avg_trade_return_pct:>+9.3f} {r.buy_hold_return_pct:>+9.2f}")
    print("=" * 140)


def plot_best_variant(df: pd.DataFrame, result: BacktestResult, save_path: str):
    """Create a 4-panel chart for the best variant."""
    params = result.params
    df_ind = add_indicators(df, params.macd_fast, params.macd_slow,
                            params.macd_signal, params.rsi_period,
                            params.ema_short, params.ema_long)

    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 2]})
    fig.suptitle(f"NVDA 30m — Best Variant: {params.short_name()} "
                 f"[{result.signal_mode}]", fontsize=13, y=0.98)

    # Panel 1: Price + EMAs + buy/sell markers
    ax1 = axes[0]
    ax1.plot(df_ind.index, df_ind["Close"], color="black", linewidth=0.6, label="Close")
    ax1.plot(df_ind.index, df_ind["EMA_Short"], color="dodgerblue", linewidth=0.5,
             label=f"EMA {params.ema_short}")
    ax1.plot(df_ind.index, df_ind["EMA_Long"], color="orange", linewidth=0.5,
             label=f"EMA {params.ema_long}")

    for t in result.trades:
        if "entry_date" in t:
            ax1.scatter(t["entry_date"], t["entry_price"], marker="^",
                        color="green", s=40, zorder=5)
        if "exit_date" in t:
            ax1.scatter(t["exit_date"], t["exit_price"], marker="v",
                        color="red", s=40, zorder=5)

    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: MACD
    ax2 = axes[1]
    ax2.plot(df_ind.index, df_ind["MACD"], color="blue", linewidth=0.6, label="MACD")
    ax2.plot(df_ind.index, df_ind["MACD_Signal"], color="red", linewidth=0.6, label="Signal")
    colors = ["green" if v >= 0 else "red" for v in df_ind["MACD_Hist"]]
    ax2.bar(df_ind.index, df_ind["MACD_Hist"], color=colors, alpha=0.4,
            width=0.01)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("MACD")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: RSI
    ax3 = axes[2]
    ax3.plot(df_ind.index, df_ind["RSI"], color="purple", linewidth=0.6)
    ax3.axhline(params.rsi_overbought, color="red", linestyle="--", linewidth=0.5)
    ax3.axhline(params.rsi_oversold, color="green", linestyle="--", linewidth=0.5)
    ax3.fill_between(df_ind.index, params.rsi_oversold, params.rsi_overbought,
                     alpha=0.05, color="gray")
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
    ax3.grid(alpha=0.3)

    # Panel 4: Equity curve vs buy & hold
    ax4 = axes[3]
    bh_equity = 100_000 * df_ind["Close"] / df_ind["Close"].iloc[0]
    ax4.plot(result.equity_curve.index, result.equity_curve, color="blue",
             linewidth=0.8, label="Strategy")
    ax4.plot(df_ind.index, bh_equity, color="gray", linewidth=0.6,
             linestyle="--", label="Buy & Hold")
    ax4.set_ylabel("Equity ($)")
    ax4.set_xlabel("Date")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {save_path}")

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  NVDA 30-Min MACD + RSI + EMA Strategy Backtester")
    print("=" * 70)

    df = fetch_nvda_30m()

    variants = build_variants()
    print(f"\nTesting {len(variants)} strategy variants...\n")

    results = []
    for params, mode in variants:
        try:
            r = backtest(df, params, signal_mode=mode)
            results.append(r)
            print(f"  {params.short_name():<28} [{mode:<8}]  "
                  f"Return: {r.total_return_pct:>+7.2f}%  "
                  f"Sharpe: {r.sharpe_ratio:>6.2f}  Trades: {r.total_trades}")
        except Exception as e:
            print(f"  {params.short_name():<28}  ERROR: {e}")

    ranked = rank_results(results)
    print_results_table(ranked)

    # Best variant details
    best = ranked[0]
    print(f"\n{'*' * 70}")
    print(f"  BEST VARIANT: {best.params.short_name()}  [{best.signal_mode}]")
    print(f"{'*' * 70}")
    print(f"  Total Return:      {best.total_return_pct:>+.2f}%")
    print(f"  Annualised Return: {best.annual_return_pct:>+.1f}%")
    print(f"  Sharpe Ratio:      {best.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:      {best.max_drawdown_pct:.2f}%")
    print(f"  Win Rate:          {best.win_rate_pct:.1f}%")
    print(f"  Total Trades:      {best.total_trades}")
    print(f"  Profit Factor:     {best.profit_factor:.2f}")
    print(f"  Avg Trade Return:  {best.avg_trade_return_pct:>+.3f}%")
    print(f"  Buy & Hold Return: {best.buy_hold_return_pct:>+.2f}%")
    print(f"\n  Parameters:")
    print(f"    MACD:        fast={best.params.macd_fast}, slow={best.params.macd_slow}, "
          f"signal={best.params.macd_signal}")
    print(f"    RSI:         period={best.params.rsi_period}, "
          f"oversold={best.params.rsi_oversold}, overbought={best.params.rsi_overbought}")
    print(f"    EMA:         short={best.params.ema_short}, long={best.params.ema_long}")
    print(f"    Stop Loss:   {best.params.stop_loss_pct:.1%}")
    print(f"    Take Profit: {best.params.take_profit_pct:.1%}")
    print(f"    Signal Mode: {best.signal_mode}")

    # Trade log
    print(f"\n  Trade Log ({best.total_trades} completed trades):")
    entries = [t for t in best.trades if "entry_date" in t]
    exits = [t for t in best.trades if "exit_date" in t]
    for i, (en, ex) in enumerate(zip(entries, exits), 1):
        print(f"    #{i:>2}  Entry: {en['entry_date']} @ ${en['entry_price']:.2f}  ->  "
              f"Exit: {ex['exit_date']} @ ${ex['exit_price']:.2f}  "
              f"P&L: {ex['pnl_pct']:>+.3f}%  ({ex['reason']})")

    # Plot
    chart_path = "/home/user/finalproject/nvda_30m_best_strategy.png"
    plot_best_variant(df, best, chart_path)

    # Save results
    summary = {
        "timeframe": "30m",
        "best_variant": best.params.short_name(),
        "signal_mode": best.signal_mode,
        "metrics": {
            "total_return_pct": round(best.total_return_pct, 3),
            "annual_return_pct": round(best.annual_return_pct, 1),
            "sharpe_ratio": round(best.sharpe_ratio, 3),
            "max_drawdown_pct": round(best.max_drawdown_pct, 3),
            "win_rate_pct": round(best.win_rate_pct, 1),
            "total_trades": best.total_trades,
            "profit_factor": round(min(best.profit_factor, 999), 2),
            "avg_trade_return_pct": round(best.avg_trade_return_pct, 3),
            "buy_hold_return_pct": round(best.buy_hold_return_pct, 2),
        },
        "parameters": {
            "macd_fast": best.params.macd_fast,
            "macd_slow": best.params.macd_slow,
            "macd_signal": best.params.macd_signal,
            "rsi_period": best.params.rsi_period,
            "rsi_oversold": best.params.rsi_oversold,
            "rsi_overbought": best.params.rsi_overbought,
            "ema_short": best.params.ema_short,
            "ema_long": best.params.ema_long,
            "stop_loss_pct": best.params.stop_loss_pct,
            "take_profit_pct": best.params.take_profit_pct,
        },
        "all_variants": [
            {
                "name": r.params.short_name(),
                "signal_mode": r.signal_mode,
                "total_return_pct": round(r.total_return_pct, 3),
                "sharpe_ratio": round(r.sharpe_ratio, 3),
                "max_drawdown_pct": round(r.max_drawdown_pct, 3),
                "win_rate": round(r.win_rate_pct, 1),
                "trades": r.total_trades,
            }
            for r in ranked
        ],
    }
    json_path = "/home/user/finalproject/backtest_results_30m.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
