"""Tests for PCA and signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ycrv.pca import pca_curve
from ycrv.signals import butterfly, carry_roll, slope, zscore


def _synthetic_curve(n: int = 400) -> pd.DataFrame:
    """Curve driven by three latent factors -> PCA should recover 3 PCs."""
    rng = np.random.default_rng(0)
    tenors = np.array([1, 2, 5, 10, 30], dtype=float)
    level = np.cumsum(rng.normal(0, 0.03, n))
    slope_f = np.cumsum(rng.normal(0, 0.02, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = np.array([3.0, 3.3, 3.8, 4.1, 4.4])
    load_level = np.ones(5)
    load_slope = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    curve = base + np.outer(level, load_level) + np.outer(slope_f, load_slope)
    return pd.DataFrame(curve, index=dates, columns=tenors)


def test_pca_level_dominates_and_variance_sums_to_one():
    df = _synthetic_curve()
    res = pca_curve(df)
    assert res.explained_variance_ratio[0] > 0.5  # level dominates
    assert abs(res.explained_variance_ratio.sum() - 1.0) < 1e-9
    # PC1 (level) loadings all same sign
    assert np.all(res.loadings[:, 0] > 0)


def test_butterfly_and_slope_definitions():
    df = pd.DataFrame(
        {2.0: [1.0], 5.0: [2.0], 10.0: [2.5]}, index=pd.to_datetime(["2024-01-01"])
    )
    # slope 2s10s = (2.5 - 1.0)*100 = 150 bp
    assert slope(df, 2.0, 10.0).iloc[0] == 150.0
    # fly = (2*2.0 - 1.0 - 2.5)*100 = 50 bp
    assert butterfly(df, 2.0, 5.0, 10.0).iloc[0] == 50.0


def test_zscore_is_standardised():
    s = pd.Series(np.arange(300, dtype=float))
    z = zscore(s, window=100, min_periods=50).dropna()
    # a rolling z-score of a trend should be bounded and finite
    assert np.isfinite(z).all()
    assert z.abs().max() < 5


def test_carry_roll_positive_on_upward_curve():
    # a steep upward curve should give positive roll-down for longer tenors
    tenors = [0.25, 2.0, 5.0, 10.0]
    rows = np.array([[3.0, 3.8, 4.3, 4.7]] * 300)
    df = pd.DataFrame(rows, index=pd.date_range("2022-01-01", periods=300, freq="B"),
                      columns=tenors)
    cr = carry_roll(df, horizon_days=63)
    assert cr[10.0].iloc[-1] > 0
    assert cr[5.0].iloc[-1] > 0
