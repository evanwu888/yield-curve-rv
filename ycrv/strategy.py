"""Butterfly mean-reversion strategy (the flagship relative-value book).

Curve butterflies are strongly mean-reverting: when the belly of a fly gets
cheap relative to its wings, it tends to snap back. The strategy trades a
basket of butterflies, sizing each by the (capped) rolling z-score of its
spread and holding every leg DV01-neutral so the book is immune to parallel
level moves (PC1) and expresses a clean view on curvature.

Sign convention (consistent with :mod:`ycrv.backtest`):

    fly spread  S = 2*y_belly - y_short - y_long
    signal      z = zscore(S)                       # high => belly cheap
    position    long the fly when z > 0
                belly DV01 = +2u,  each wing DV01 = -u,   u = f(z) > 0

A long fly profits when S falls (the belly richens back), so fading a high
z-score is the mean-reversion bet. The three legs sum to zero DV01.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .signals import butterfly, carry_roll, zscore


@dataclass
class Fly:
    short: float
    belly: float
    long: float

    def __post_init__(self) -> None:
        if not (self.short < self.belly < self.long):
            raise ValueError(f"Need short < belly < long, got {self}")

    @property
    def name(self) -> str:
        def fmt(t: float) -> str:
            return f"{t:g}y" if t >= 1 else f"{t*12:g}m"

        return f"{fmt(self.short)}-{fmt(self.belly)}-{fmt(self.long)}"


DEFAULT_FLIES: list[Fly] = [
    Fly(0.5, 2.0, 5.0),
    Fly(2.0, 5.0, 10.0),
    Fly(5.0, 10.0, 30.0),
]


@dataclass
class ButterflyRVStrategy:
    flies: list[Fly] = field(default_factory=lambda: list(DEFAULT_FLIES))
    z_window: int = 120
    z_min_periods: int = 60
    z_cap: float = 2.5
    target_dv01: float = 100.0  # DV01 per wing at a full-strength (|z|=cap) signal
    rebalance_days: int = 1  # re-trade cadence; >1 holds the book between marks
    entry_band: float = 0.0  # optional no-trade band on |z| to suppress churn

    def spreads(self, yields: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {f.name: butterfly(yields, f.short, f.belly, f.long) for f in self.flies}
        )

    def signals(self, yields: pd.DataFrame) -> pd.DataFrame:
        """Per-fly z-scores (the trading signals)."""
        spr = self.spreads(yields)
        return spr.apply(lambda s: zscore(s, self.z_window, self.z_min_periods))

    def positions(self, yields: pd.DataFrame) -> pd.DataFrame:
        """Aggregate signed dollar-DV01 per tenor across all flies.

        Turnover is controlled two ways, exactly as a desk would: a no-trade
        band suppresses tiny signals, and the book is only re-marked every
        ``rebalance_days`` (held flat in between).
        """
        z = self.signals(yields)
        pos = pd.DataFrame(0.0, index=yields.index, columns=yields.columns)
        for f in self.flies:
            zc = z[f.name].clip(-self.z_cap, self.z_cap)
            # no-trade band: zero out weak signals
            zc = zc.where(zc.abs() >= self.entry_band, 0.0)
            # u > 0 means long the fly (long belly, short wings)
            u = self.target_dv01 * (zc / self.z_cap)
            pos[f.belly] = pos[f.belly].add(2.0 * u, fill_value=0.0)
            pos[f.short] = pos[f.short].add(-u, fill_value=0.0)
            pos[f.long] = pos[f.long].add(-u, fill_value=0.0)
        pos = pos.fillna(0.0)

        # Hold the book between rebalance dates (step-wise, no daily churn):
        # blank out non-rebalance rows, then forward-fill the last traded book.
        if self.rebalance_days > 1:
            keep = np.zeros(len(pos), dtype=bool)
            keep[:: self.rebalance_days] = True
            blanked = pos.copy()
            blanked.iloc[~keep] = np.nan
            pos = blanked.ffill().fillna(0.0)
        return pos

    def run(self, yields: pd.DataFrame, half_spread_bp: float = 0.1) -> BacktestResult:
        return run_backtest(self.positions(yields), yields, half_spread_bp)

    def carry_roll_snapshot(
        self, yields: pd.DataFrame, horizon_days: int = 63
    ) -> pd.Series:
        """Latest expected carry + roll-down (bp) per tenor -- an analytics view."""
        cr = carry_roll(yields, horizon_days=horizon_days)
        return cr.iloc[-1]
