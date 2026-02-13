"""
NVDA MACD + RSI + EMA Combined Trading Strategy
=================================================
Fetches NVDA 1D data, computes MACD, RSI, and EMA indicators,
backtests multiple parameter variants, and reports the best one.
"""

import itertools
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_nvda_data(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download NVDA daily OHLCV data from Yahoo Finance."""
    ticker = yf.Ticker("NVDA")
    df = ticker.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"Fetched {len(df)} daily bars for NVDA  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    return df

# ---------------------------------------------------------------------------
# 2. TECHNICAL INDICATORS
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
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


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
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
    """All tuneable parameters for the MACD+RSI+EMA strategy."""
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    # EMA crossover
    ema_short: int = 20
    ema_long: int = 50
    # Risk management
    stop_loss_pct: float = 0.05       # 5 % trailing stop-loss
    take_profit_pct: float = 0.15     # 15 % take-profit
    # Label
    name: str = ""

    def short_name(self) -> str:
        if self.name:
            return self.name
        return (f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})"
                f"_RSI({self.rsi_period},{self.rsi_oversold:.0f},{self.rsi_overbought:.0f})"
                f"_EMA({self.ema_short},{self.ema_long})"
                f"_SL{self.stop_loss_pct:.0%}_TP{self.take_profit_pct:.0%}")


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """
    Combined entry logic (all three must agree):
      BUY  when MACD line crosses ABOVE signal AND RSI < oversold threshold
             AND EMA_Short > EMA_Long  (trend confirmation)
      SELL when MACD line crosses BELOW signal AND RSI > overbought threshold
             AND EMA_Short < EMA_Long
    """
    df = df.copy()

    # MACD crossover detection
    df["MACD_Cross_Up"] = (df["MACD"] > df["MACD_Signal"]) & \
                          (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    df["MACD_Cross_Down"] = (df["MACD"] < df["MACD_Signal"]) & \
                            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

    # RSI zones
    df["RSI_Oversold"] = df["RSI"] < params.rsi_oversold
    df["RSI_Overbought"] = df["RSI"] > params.rsi_overbought

    # EMA trend
    df["EMA_Bullish"] = df["EMA_Short"] > df["EMA_Long"]
    df["EMA_Bearish"] = df["EMA_Short"] < df["EMA_Long"]

    # Combined signals  (1 = BUY, -1 = SELL, 0 = hold)
    df["Signal"] = 0
    df.loc[df["MACD_Cross_Up"] & df["RSI_Oversold"] & df["EMA_Bullish"], "Signal"] = 1
    df.loc[df["MACD_Cross_Down"] & df["RSI_Overbought"] & df["EMA_Bearish"], "Signal"] = -1

    return df


def generate_signals_relaxed(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """
    Relaxed entry logic (2-of-3 confirmation):
      BUY  when at least 2 of: MACD cross up, RSI recovering from oversold zone,
             EMA bullish crossover
      SELL when at least 2 of the bearish equivalents
    """
    df = df.copy()

    # MACD crossover
    df["MACD_Cross_Up"] = (df["MACD"] > df["MACD_Signal"]) & \
                          (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    df["MACD_Cross_Down"] = (df["MACD"] < df["MACD_Signal"]) & \
                            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

    # RSI zones (use a wider band for relaxed)
    df["RSI_Oversold"] = df["RSI"] < params.rsi_oversold + 10   # e.g. < 40
    df["RSI_Overbought"] = df["RSI"] > params.rsi_overbought - 10  # e.g. > 60

    # EMA trend
    df["EMA_Bullish"] = df["EMA_Short"] > df["EMA_Long"]
    df["EMA_Bearish"] = df["EMA_Short"] < df["EMA_Long"]

    # Score
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

# ---------------------------------------------------------------------------
# 4. BACKTESTING ENGINE
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    params: StrategyParams
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
             signal_mode: str = "strict") -> BacktestResult:
    """
    Run a vectorised + event-driven backtest.
    Position sizing: 100 % of equity per trade (long only).
    """
    # Compute indicators
    df_ind = add_indicators(df, params.macd_fast, params.macd_slow,
                            params.macd_signal, params.rsi_period,
                            params.ema_short, params.ema_long)

    # Generate signals
    if signal_mode == "relaxed":
        df_sig = generate_signals_relaxed(df_ind, params)
    else:
        df_sig = generate_signals(df_ind, params)

    # Walk through bars
    capital = initial_capital
    position = 0          # shares held
    entry_price = 0.0
    high_water = 0.0      # for trailing stop
    equity = []
    trades = []

    for i in range(len(df_sig)):
        row = df_sig.iloc[i]
        price = row["Close"]
        signal = row["Signal"]
        date = df_sig.index[i]

        # If in position, check stop-loss / take-profit
        if position > 0:
            current_val = position * price
            high_water = max(high_water, current_val)
            drawdown_from_peak = (high_water - current_val) / high_water

            ret_from_entry = (price - entry_price) / entry_price

            # Trailing stop-loss or take-profit
            if drawdown_from_peak >= params.stop_loss_pct or \
               ret_from_entry >= params.take_profit_pct or \
               signal == -1:
                # SELL
                capital = position * price
                pnl_pct = (price - entry_price) / entry_price * 100
                trades.append({
                    "exit_date": date,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "reason": ("SL" if drawdown_from_peak >= params.stop_loss_pct
                               else "TP" if ret_from_entry >= params.take_profit_pct
                               else "Signal"),
                })
                position = 0
                entry_price = 0.0
                high_water = 0.0

        # Entry
        if position == 0 and signal == 1:
            position = capital / price
            entry_price = price
            high_water = capital
            capital = 0.0
            trades.append({"entry_date": date, "entry_price": price})

        # Track equity
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

    # Metrics
    final_equity = equity[-1] if equity else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100

    n_years = (df_sig.index[-1] - df_sig.index[0]).days / 365.25
    annual_return = ((final_equity / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100

    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)

    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_dd = drawdowns.min() * 100

    # Trade stats
    completed = [t for t in trades if "pnl_pct" in t]
    wins = [t for t in completed if t["pnl_pct"] > 0]
    losses = [t for t in completed if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(completed) * 100 if completed else 0
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_trade_ret = np.mean([t["pnl_pct"] for t in completed]) if completed else 0

    # Buy & hold benchmark
    bh_return = (df_sig.iloc[-1]["Close"] / df_sig.iloc[0]["Close"] - 1) * 100

    return BacktestResult(
        params=params,
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
# 5. PARAMETER VARIANTS
# ---------------------------------------------------------------------------

def build_variants() -> list[tuple[StrategyParams, str]]:
    """Return a list of (StrategyParams, signal_mode) variants to test."""
    variants = []

    # --- Strict signal mode (all 3 indicators must agree) ---

    # V1: Classic defaults
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        ema_short=20, ema_long=50,
        stop_loss_pct=0.05, take_profit_pct=0.15,
        name="V1_Classic_Strict",
    ), "strict"))

    # V2: Faster MACD with tighter RSI
    variants.append((StrategyParams(
        macd_fast=8, macd_slow=21, macd_signal=5,
        rsi_period=10, rsi_oversold=35, rsi_overbought=65,
        ema_short=10, ema_long=30,
        stop_loss_pct=0.04, take_profit_pct=0.12,
        name="V2_Fast_Strict",
    ), "strict"))

    # V3: Slow/conservative
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=21, rsi_oversold=25, rsi_overbought=75,
        ema_short=50, ema_long=100,
        stop_loss_pct=0.07, take_profit_pct=0.20,
        name="V3_Conservative_Strict",
    ), "strict"))

    # V4: Medium with wider SL/TP
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        ema_short=20, ema_long=50,
        stop_loss_pct=0.08, take_profit_pct=0.25,
        name="V4_WideSLTP_Strict",
    ), "strict"))

    # V5: Tight with fast EMA
    variants.append((StrategyParams(
        macd_fast=8, macd_slow=17, macd_signal=9,
        rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        ema_short=9, ema_long=21,
        stop_loss_pct=0.03, take_profit_pct=0.10,
        name="V5_Tight_Strict",
    ), "strict"))

    # --- Relaxed signal mode (2-of-3 indicators) ---

    # V6: Classic relaxed
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        ema_short=20, ema_long=50,
        stop_loss_pct=0.05, take_profit_pct=0.15,
        name="V6_Classic_Relaxed",
    ), "relaxed"))

    # V7: Fast relaxed
    variants.append((StrategyParams(
        macd_fast=8, macd_slow=21, macd_signal=5,
        rsi_period=10, rsi_oversold=35, rsi_overbought=65,
        ema_short=10, ema_long=30,
        stop_loss_pct=0.04, take_profit_pct=0.12,
        name="V7_Fast_Relaxed",
    ), "relaxed"))

    # V8: Conservative relaxed
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=21, rsi_oversold=25, rsi_overbought=75,
        ema_short=50, ema_long=100,
        stop_loss_pct=0.07, take_profit_pct=0.20,
        name="V8_Conservative_Relaxed",
    ), "relaxed"))

    # V9: Wide SL/TP relaxed
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        ema_short=20, ema_long=50,
        stop_loss_pct=0.08, take_profit_pct=0.25,
        name="V9_WideSLTP_Relaxed",
    ), "relaxed"))

    # V10: Aggressive relaxed – small SL, generous TP
    variants.append((StrategyParams(
        macd_fast=8, macd_slow=17, macd_signal=9,
        rsi_period=14, rsi_oversold=35, rsi_overbought=65,
        ema_short=9, ema_long=21,
        stop_loss_pct=0.03, take_profit_pct=0.20,
        name="V10_Aggressive_Relaxed",
    ), "relaxed"))

    # V11: Trend-following relaxed – very wide EMA, large TP
    variants.append((StrategyParams(
        macd_fast=12, macd_slow=26, macd_signal=9,
        rsi_period=14, rsi_oversold=40, rsi_overbought=60,
        ema_short=20, ema_long=50,
        stop_loss_pct=0.06, take_profit_pct=0.30,
        name="V11_TrendFollow_Relaxed",
    ), "relaxed"))

    # V12: Momentum relaxed – fast indicators, medium risk
    variants.append((StrategyParams(
        macd_fast=5, macd_slow=13, macd_signal=4,
        rsi_period=7, rsi_oversold=30, rsi_overbought=70,
        ema_short=8, ema_long=21,
        stop_loss_pct=0.05, take_profit_pct=0.15,
        name="V12_Momentum_Relaxed",
    ), "relaxed"))

    return variants

# ---------------------------------------------------------------------------
# 6. REPORTING & CHARTING
# ---------------------------------------------------------------------------

def print_results_table(results: list[BacktestResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 130)
    print(f"{'Variant':<30} {'Return%':>9} {'Annual%':>9} {'Sharpe':>8} "
          f"{'MaxDD%':>9} {'WinRate%':>9} {'Trades':>7} {'PF':>7} "
          f"{'AvgTrade%':>10} {'B&H%':>9}")
    print("=" * 130)
    for r in results:
        print(f"{r.params.short_name():<30} {r.total_return_pct:>+9.1f} "
              f"{r.annual_return_pct:>+9.1f} {r.sharpe_ratio:>8.2f} "
              f"{r.max_drawdown_pct:>9.1f} {r.win_rate_pct:>9.1f} "
              f"{r.total_trades:>7} {r.profit_factor:>7.2f} "
              f"{r.avg_trade_return_pct:>+10.2f} {r.buy_hold_return_pct:>+9.1f}")
    print("=" * 130)


def plot_best_variant(df: pd.DataFrame, result: BacktestResult, save_path: str):
    """Create a 4-panel chart for the best variant."""
    params = result.params
    df_ind = add_indicators(df, params.macd_fast, params.macd_slow,
                            params.macd_signal, params.rsi_period,
                            params.ema_short, params.ema_long)

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 2]})
    fig.suptitle(f"NVDA 1D — Best Variant: {params.short_name()}", fontsize=14, y=0.98)

    # --- Panel 1: Price + EMAs + buy/sell markers ---
    ax1 = axes[0]
    ax1.plot(df_ind.index, df_ind["Close"], color="black", linewidth=0.8, label="Close")
    ax1.plot(df_ind.index, df_ind["EMA_Short"], color="dodgerblue", linewidth=0.7,
             label=f"EMA {params.ema_short}")
    ax1.plot(df_ind.index, df_ind["EMA_Long"], color="orange", linewidth=0.7,
             label=f"EMA {params.ema_long}")

    # Buy/sell markers from trades
    for t in result.trades:
        if "entry_date" in t:
            ax1.annotate("BUY", xy=(t["entry_date"], t["entry_price"]),
                         fontsize=7, color="green", fontweight="bold",
                         ha="center", va="top")
            ax1.scatter(t["entry_date"], t["entry_price"], marker="^",
                        color="green", s=60, zorder=5)
        if "exit_date" in t:
            ax1.scatter(t["exit_date"], t["exit_price"], marker="v",
                        color="red", s=60, zorder=5)

    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Panel 2: MACD ---
    ax2 = axes[1]
    ax2.plot(df_ind.index, df_ind["MACD"], color="blue", linewidth=0.7, label="MACD")
    ax2.plot(df_ind.index, df_ind["MACD_Signal"], color="red", linewidth=0.7, label="Signal")
    colors = ["green" if v >= 0 else "red" for v in df_ind["MACD_Hist"]]
    ax2.bar(df_ind.index, df_ind["MACD_Hist"], color=colors, alpha=0.4, width=1)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("MACD")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # --- Panel 3: RSI ---
    ax3 = axes[2]
    ax3.plot(df_ind.index, df_ind["RSI"], color="purple", linewidth=0.7)
    ax3.axhline(params.rsi_overbought, color="red", linestyle="--", linewidth=0.5)
    ax3.axhline(params.rsi_oversold, color="green", linestyle="--", linewidth=0.5)
    ax3.fill_between(df_ind.index, params.rsi_oversold, params.rsi_overbought,
                     alpha=0.05, color="gray")
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
    ax3.grid(alpha=0.3)

    # --- Panel 4: Equity curve vs buy & hold ---
    ax4 = axes[3]
    bh_equity = 100_000 * df_ind["Close"] / df_ind["Close"].iloc[0]
    ax4.plot(result.equity_curve.index, result.equity_curve, color="blue",
             linewidth=1.0, label="Strategy")
    ax4.plot(df_ind.index, bh_equity, color="gray", linewidth=0.8,
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

def rank_results(results: list[BacktestResult]) -> list[BacktestResult]:
    """
    Rank by a composite score:
      score = 0.30*norm(annual_return) + 0.25*norm(sharpe) +
              0.20*norm(-max_dd) + 0.15*norm(win_rate) + 0.10*norm(profit_factor)
    """
    if not results:
        return results

    def safe_norm(vals):
        arr = np.array(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn == 0:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    ann = [r.annual_return_pct for r in results]
    sha = [r.sharpe_ratio for r in results]
    dd = [-r.max_drawdown_pct for r in results]  # higher (less negative) is better
    wr = [r.win_rate_pct for r in results]
    pf = [min(r.profit_factor, 10) for r in results]  # cap inf

    n_ann = safe_norm(ann)
    n_sha = safe_norm(sha)
    n_dd = safe_norm(dd)
    n_wr = safe_norm(wr)
    n_pf = safe_norm(pf)

    scores = 0.30 * n_ann + 0.25 * n_sha + 0.20 * n_dd + 0.15 * n_wr + 0.10 * n_pf

    paired = list(zip(scores, results))
    paired.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in paired]


def main():
    print("=" * 70)
    print("  NVDA MACD + RSI + EMA Strategy Backtester")
    print("=" * 70)

    # Fetch data
    df = fetch_nvda_data(period="5y", interval="1d")

    # Build variants
    variants = build_variants()
    print(f"\nTesting {len(variants)} strategy variants...\n")

    # Run backtests
    results = []
    for params, mode in variants:
        try:
            r = backtest(df, params, signal_mode=mode)
            results.append(r)
            print(f"  {params.short_name():<30}  Return: {r.total_return_pct:>+8.1f}%  "
                  f"Sharpe: {r.sharpe_ratio:.2f}  Trades: {r.total_trades}")
        except Exception as e:
            print(f"  {params.short_name():<30}  ERROR: {e}")

    # Rank and display
    ranked = rank_results(results)
    print_results_table(ranked)

    # Best variant details
    best = ranked[0]
    print(f"\n{'*' * 70}")
    print(f"  BEST VARIANT: {best.params.short_name()}")
    print(f"{'*' * 70}")
    print(f"  Total Return:      {best.total_return_pct:>+.2f}%")
    print(f"  Annualised Return: {best.annual_return_pct:>+.2f}%")
    print(f"  Sharpe Ratio:      {best.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:      {best.max_drawdown_pct:.2f}%")
    print(f"  Win Rate:          {best.win_rate_pct:.1f}%")
    print(f"  Total Trades:      {best.total_trades}")
    print(f"  Profit Factor:     {best.profit_factor:.2f}")
    print(f"  Avg Trade Return:  {best.avg_trade_return_pct:>+.2f}%")
    print(f"  Buy & Hold Return: {best.buy_hold_return_pct:>+.2f}%")
    print(f"\n  Parameters:")
    print(f"    MACD:       fast={best.params.macd_fast}, slow={best.params.macd_slow}, "
          f"signal={best.params.macd_signal}")
    print(f"    RSI:        period={best.params.rsi_period}, "
          f"oversold={best.params.rsi_oversold}, overbought={best.params.rsi_overbought}")
    print(f"    EMA:        short={best.params.ema_short}, long={best.params.ema_long}")
    print(f"    Stop Loss:  {best.params.stop_loss_pct:.0%}")
    print(f"    Take Profit:{best.params.take_profit_pct:.0%}")

    # Trade log
    print(f"\n  Trade Log ({best.total_trades} completed trades):")
    entries = [t for t in best.trades if "entry_date" in t]
    exits = [t for t in best.trades if "exit_date" in t]
    for i, (en, ex) in enumerate(zip(entries, exits), 1):
        print(f"    #{i:>2}  Entry: {en['entry_date'].date()} @ ${en['entry_price']:.2f}  ->  "
              f"Exit: {ex['exit_date'].date()} @ ${ex['exit_price']:.2f}  "
              f"P&L: {ex['pnl_pct']:>+.2f}%  ({ex['reason']})")

    # Plot
    plot_best_variant(df, best, "/home/user/finalproject/nvda_best_strategy.png")

    # Save results summary to JSON
    summary = {
        "best_variant": best.params.short_name(),
        "metrics": {
            "total_return_pct": round(best.total_return_pct, 2),
            "annual_return_pct": round(best.annual_return_pct, 2),
            "sharpe_ratio": round(best.sharpe_ratio, 3),
            "max_drawdown_pct": round(best.max_drawdown_pct, 2),
            "win_rate_pct": round(best.win_rate_pct, 1),
            "total_trades": best.total_trades,
            "profit_factor": round(min(best.profit_factor, 999), 2),
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
        "signal_mode": "relaxed" if "Relaxed" in best.params.short_name() else "strict",
        "all_variants": [
            {
                "name": r.params.short_name(),
                "total_return_pct": round(r.total_return_pct, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 3),
                "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                "trades": r.total_trades,
            }
            for r in ranked
        ],
    }
    with open("/home/user/finalproject/backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to backtest_results.json")


if __name__ == "__main__":
    main()
