"""Relative-value signals on the curve.

Three families, all returning tidy pandas objects:

* **Spreads** -- slopes (e.g. 2s10s) and butterflies (e.g. 2s5s10s), the raw
  curve tradables.
* **Mean reversion** -- rolling z-scores of those spreads. Curve spreads are
  strongly mean-reverting, so a high z-score is a signal to fade.
* **Carry & roll-down** -- the expected P&L of simply holding a bond if the
  curve does not move, decomposed into coupon carry vs. rolling down the curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def slope(yields: pd.DataFrame, short: float, long: float) -> pd.Series:
    """Curve slope long - short (bp), e.g. 2s10s = y(10) - y(2)."""
    return (yields[long] - yields[short]) * 100.0


def butterfly(
    yields: pd.DataFrame, short: float, belly: float, long: float
) -> pd.Series:
    """Butterfly spread 2*belly - short - long (bp).

    A rising fly means the belly is cheapening relative to the wings.
    """
    return (2.0 * yields[belly] - yields[short] - yields[long]) * 100.0


def zscore(series: pd.Series, window: int = 120, min_periods: int = 60) -> pd.Series:
    """Rolling z-score, (x - rolling_mean) / rolling_std."""
    roll = series.rolling(window, min_periods=min_periods)
    return (series - roll.mean()) / roll.std(ddof=1)


def interp_curve(curve_row: pd.Series, tenor: float) -> float:
    """Linearly interpolate a single day's curve at an arbitrary tenor."""
    tenors = curve_row.index.to_numpy(dtype=float)
    vals = curve_row.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    return float(np.interp(tenor, tenors[finite], vals[finite]))


def carry_roll(
    yields: pd.DataFrame, horizon_days: int = 63, funding_tenor: float = 0.25
) -> pd.DataFrame:
    """Carry + roll-down per tenor over a holding horizon, in basis points.

    For each date and tenor T we compute the expected excess return if the curve
    is unchanged over the horizon ``h`` (in years):

        roll_down(bp) = y(T) - y(T - h)      # yield falls as the bond ages,
                                             # so an upward curve => positive roll
        carry(bp)     = (y(T) - funding) * h # coupon income net of funding

    Returned as a DataFrame (dates x tenors) of carry + roll in bp. This is a
    directional cross-sectional signal: high carry+roll tenors are attractive to
    own, and it can tilt a mean-reversion book.
    """
    h_years = horizon_days / 252.0
    tenors = np.asarray(yields.columns, dtype=float)
    out = pd.DataFrame(index=yields.index, columns=yields.columns, dtype=float)

    for date, row in yields.iterrows():
        funding = interp_curve(row, funding_tenor)
        for T in tenors:
            rolled_tenor = max(T - h_years, tenors.min())
            y_now = interp_curve(row, T)
            y_rolled = interp_curve(row, rolled_tenor)
            roll_bp = (y_now - y_rolled) * 100.0
            carry_bp = (y_now - funding) * h_years * 100.0
            out.loc[date, T] = carry_bp + roll_bp
    return out
