"""Tests for the backtester and strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ycrv.backtest import run_backtest
from ycrv.strategy import ButterflyRVStrategy, Fly


def test_long_position_loses_when_yields_rise():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    yields = pd.DataFrame({10.0: [4.0, 4.1, 4.2]}, index=dates)  # +10bp/day
    positions = pd.DataFrame({10.0: [100.0, 100.0, 100.0]}, index=dates)  # long DV01
    res = run_backtest(positions, yields, half_spread_bp=0.0)
    # long 100 DV01, yield +10bp -> lose 100*10 = 1000 per step
    assert np.allclose(res.pnl.to_numpy(), [-1000.0, -1000.0])


def test_dv01_neutral_book_immune_to_parallel_shift():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    # parallel +5bp/day shift across the whole curve
    yields = pd.DataFrame(
        {
            2.0: [4.00, 4.05, 4.10, 4.15],
            5.0: [4.30, 4.35, 4.40, 4.45],
            10.0: [4.60, 4.65, 4.70, 4.75],
        },
        index=dates,
    )
    # DV01-neutral fly: +2 belly, -1 each wing
    positions = pd.DataFrame(
        {2.0: [-50.0] * 4, 5.0: [100.0] * 4, 10.0: [-50.0] * 4}, index=dates
    )
    res = run_backtest(positions, yields, half_spread_bp=0.0)
    # pure parallel shift -> a DV01-neutral book has ~zero PnL
    assert np.allclose(res.pnl.to_numpy(), 0.0, atol=1e-9)


def test_strategy_positions_are_dv01_neutral_each_day():
    strat = ButterflyRVStrategy(flies=[Fly(2.0, 5.0, 10.0)])
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    y = 4 + np.cumsum(rng.normal(0, 0.02, (200, 3)), axis=0)
    yields = pd.DataFrame(y, index=dates, columns=[2.0, 5.0, 10.0])
    pos = strat.positions(yields)
    # each leg sums to zero DV01 across the trade (immune to level)
    assert np.allclose(pos.sum(axis=1).to_numpy(), 0.0, atol=1e-9)


def test_no_lookahead_first_pnl_is_nan_dropped():
    strat = ButterflyRVStrategy(flies=[Fly(2.0, 5.0, 10.0)], z_min_periods=10)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(2)
    y = 4 + np.cumsum(rng.normal(0, 0.02, (100, 3)), axis=0)
    yields = pd.DataFrame(y, index=dates, columns=[2.0, 5.0, 10.0])
    res = strat.run(yields)
    # backtest drops the first (undefined) return; PnL never precedes signal
    assert res.pnl.index[0] > yields.index[0]
