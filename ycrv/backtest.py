"""DV01-space backtester.

Positions are expressed as **signed dollar-DV01 per tenor**: a value of +100
means "long enough of that tenor to make $100 for a 1bp fall in its yield"
(long duration). This is exactly how a rates desk books a curve trade, and it
makes DV01-neutrality a simple linear constraint (the row of positions sums to
zero across the legs of the trade).

Daily P&L for a position held from t to t+1 is

    pnl_{t+1} = - sum_tenor  position_t(tenor) * dy_{t+1}(tenor_in_bp)

i.e. a long position (positive DV01) loses money when yields rise. Transaction
costs are charged on the change in position, using a per-leg bid/ask
half-spread expressed in bp of yield.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import PerfStats, perf_stats


@dataclass
class BacktestResult:
    pnl: pd.Series  # daily P&L, dollars
    equity: pd.Series  # cumulative P&L
    turnover: pd.Series  # daily |change in DV01|, dollars
    costs: pd.Series  # daily transaction cost, dollars
    positions: pd.DataFrame
    stats: PerfStats

    def summary(self) -> str:  # pragma: no cover - cosmetic
        gross = self.turnover.mean()
        return (
            f"{self.stats}\n"
            f"  Avg turnover  : {gross:>12,.0f}  (DV01/day)\n"
            f"  Total costs   : {self.costs.sum():>12,.0f}"
        )


def run_backtest(
    positions: pd.DataFrame,
    yields: pd.DataFrame,
    half_spread_bp: float = 0.1,
) -> BacktestResult:
    """Backtest a stream of DV01 positions against realised yield changes.

    Parameters
    ----------
    positions: DataFrame (dates x tenors) of signed dollar-DV01. The position on
        date ``t`` is assumed to be established at that day's close and to earn
        the yield change into ``t+1`` (no look-ahead).
    yields: DataFrame (dates x tenors) of yields in percent.
    half_spread_bp: one-way bid/ask half-spread in bp charged on DV01 traded.
    """
    positions = positions.sort_index()
    yields = yields.sort_index().reindex(columns=positions.columns)

    # yield changes in bp, aligned so pnl on date t uses position from t-1
    dy_bp = yields.diff() * 100.0
    pos_lag = positions.shift(1)

    # min_count=1 keeps all-NaN rows (e.g. the first day, which has no prior
    # position) as NaN rather than a spurious 0 -- this is what enforces the
    # no-look-ahead property.
    gross_pnl = -(pos_lag * dy_bp).sum(axis=1, min_count=1)

    # transaction cost on the change in DV01 position
    dpos = positions.diff().abs()
    turnover = dpos.sum(axis=1, min_count=1)
    costs = turnover * half_spread_bp

    pnl = (gross_pnl - costs).dropna()
    equity = pnl.cumsum()

    return BacktestResult(
        pnl=pnl,
        equity=equity,
        turnover=turnover.reindex(pnl.index).fillna(0.0),
        costs=costs.reindex(pnl.index).fillna(0.0),
        positions=positions,
        stats=perf_stats(pnl),
    )
