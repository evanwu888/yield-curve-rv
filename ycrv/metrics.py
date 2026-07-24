"""Performance statistics for a daily PnL stream.

Everything is computed from a daily PnL series expressed in dollars (or any
consistent unit). Sharpe / Sortino / hit-rate are scale-invariant, so the
absolute dollar scale of the strategy does not affect them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerfStats:
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    skew: float
    n_days: int

    def as_dict(self) -> dict[str, float]:
        return {
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "skew": self.skew,
            "n_days": self.n_days,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"  Ann. return   : {self.ann_return:>12,.0f}\n"
            f"  Ann. vol      : {self.ann_vol:>12,.0f}\n"
            f"  Sharpe        : {self.sharpe:>12.2f}\n"
            f"  Sortino       : {self.sortino:>12.2f}\n"
            f"  Max drawdown  : {self.max_drawdown:>12,.0f}\n"
            f"  Calmar        : {self.calmar:>12.2f}\n"
            f"  Hit rate      : {self.hit_rate:>12.1%}\n"
            f"  Daily skew    : {self.skew:>12.2f}\n"
            f"  Days          : {self.n_days:>12,d}"
        )


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drop of a cumulative-PnL (equity) curve."""
    running_max = equity.cummax()
    drawdown = equity - running_max
    return float(drawdown.min())


def perf_stats(pnl: pd.Series) -> PerfStats:
    """Summary statistics for a daily PnL series."""
    pnl = pnl.dropna()
    if len(pnl) == 0:
        raise ValueError("Empty PnL series.")

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))
    downside = pnl[pnl < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan

    ann_return = mean * TRADING_DAYS
    ann_vol = std * np.sqrt(TRADING_DAYS)
    sharpe = (mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    sortino = (
        mean / downside_std * np.sqrt(TRADING_DAYS)
        if downside_std and downside_std > 0
        else np.nan
    )

    equity = pnl.cumsum()
    mdd = max_drawdown(equity)
    calmar = (ann_return / abs(mdd)) if mdd < 0 else np.nan
    hit_rate = float((pnl > 0).mean())
    skew = float(pnl.skew())

    return PerfStats(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        hit_rate=hit_rate,
        skew=skew,
        n_days=len(pnl),
    )
