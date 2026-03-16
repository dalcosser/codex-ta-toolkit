import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    stats: dict  # all plain floats (not Series) for easy UI formatting

def vectorized_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    price_col: str = "Close",
    fee_bps: float = 0.0,
) -> BacktestResult:
    """
    Vectorized long/flat backtest.
    - signal ∈ {-1, 0, 1} or {0,1}
    - fee_bps charged on position changes (turns)
    """
    if price_col not in data.columns:
        raise ValueError(f"price_col '{price_col}' not in data columns")

    df = data.copy()
    sig = signal.reindex(df.index).fillna(0).clip(-1, 1)

    # Position = yesterday's signal
    pos = sig.shift(1).fillna(0)

    # Returns
    ret = df[price_col].pct_change().fillna(0.0)
    strat_ret = pos * ret

    # Fees on turns
    turns = (pos - pos.shift(1).fillna(0)).abs()
    fee = turns * (fee_bps / 1e4)
    strat_ret_after_fee = strat_ret - fee

    equity = (1.0 + strat_ret_after_fee).cumprod()

    # Trade log
    entries = (turns > 0) & (pos != 0)
    exits   = (turns > 0) & (pos == 0)
    trade_log = []
    current_entry_price = None
    current_entry_date = None

    px = df[price_col]
    for date, is_entry, is_exit, p, position in zip(df.index, entries, exits, px, pos):
        if is_entry and current_entry_price is None and position != 0:
            current_entry_price = float(p)
            current_entry_date = date
        elif is_exit and current_entry_price is not None:
            trade_log.append(
                {
                    "entry_date": current_entry_date,
                    "exit_date": date,
                    "entry_price": float(current_entry_price),
                    "exit_price": float(p),
                    "return_pct": float(p / current_entry_price - 1.0),
                }
            )
            current_entry_price = None
            current_entry_date = None

    trades = pd.DataFrame(trade_log)

    # ----- stats as plain floats -----
    n = len(equity)
    total_return = float((equity.iloc[-1] - 1.0) * 100.0) if n else 0.0
    cagr = float(((equity.iloc[-1]) ** (252.0 / n) - 1.0) * 100.0) if n else 0.0
    mdd = float(((equity / equity.cummax()).min() - 1.0) * 100.0) if n else 0.0
    sharpe = float((strat_ret_after_fee.mean() / (strat_ret_after_fee.std() + 1e-12)) * np.sqrt(252.0)) if n else 0.0
    win_rate = float((trades["return_pct"] > 0).mean() * 100.0) if not trades.empty else None

    stats = {
        "Total_Return_%": total_return,
        "CAGR_%": cagr,
        "Max_Drawdown_%": mdd,
        "Sharpe": sharpe,
        "Win_Rate_%": win_rate,
        "Num_Trades": int(len(trades)),
    }

    return BacktestResult(equity_curve=equity, trades=trades, stats=stats)

