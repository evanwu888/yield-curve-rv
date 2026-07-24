#!/usr/bin/env python3
"""End-to-end research report for the yield-curve RV toolkit.

Runs the full pipeline on the bundled Treasury history and prints a desk-style
report:

  1. Curve snapshot + NSS fit and rich/cheap residuals
  2. PCA factor decomposition (level / slope / curvature)
  3. Carry + roll-down table for the latest curve
  4. Butterfly mean-reversion backtest with performance stats
  5. Transaction-cost sensitivity

Optionally writes charts to docs/ when matplotlib is installed:

    python scripts/run_research.py            # console report
    python scripts/run_research.py --charts   # also save PNGs to docs/
    python scripts/run_research.py --live     # fetch fresh data from FRED
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# allow `python scripts/run_research.py` from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ycrv.curve import fit_nss
from ycrv.data import fetch_fred, load_sample
from ycrv.pca import pca_curve
from ycrv.strategy import ButterflyRVStrategy

warnings.filterwarnings("ignore")
DOCS = Path(__file__).resolve().parent.parent / "docs"


def hdr(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def tenor_label(t: float) -> str:
    return f"{t:g}y" if t >= 1 else f"{t*12:g}m"


def report(df: pd.DataFrame, make_charts: bool) -> None:
    print(
        f"Loaded {len(df):,} daily curves, "
        f"{df.index.min().date()} -> {df.index.max().date()}, "
        f"{df.shape[1]} tenors."
    )

    # 1. Curve + NSS fit ---------------------------------------------------
    hdr("1. LATEST CURVE & NSS FIT (rich / cheap)")
    row = df.iloc[-1]
    fit = fit_nss(row.index.to_numpy(), row.to_numpy())
    print(f"As of {df.index[-1].date()}   NSS RMSE = {fit.rmse*100:.1f} bp\n")
    print(f"  {'tenor':>6} {'yield':>7} {'fitted':>7} {'resid(bp)':>10}  rich/cheap")
    resid_bp = fit.residuals * 100.0
    for t, y, fv, rb in zip(fit.tenors, fit.observed, fit.fitted, resid_bp):
        tag = "CHEAP" if rb > 1 else ("rich" if rb < -1 else "")
        print(f"  {tenor_label(t):>6} {y:>7.2f} {fv:>7.2f} {rb:>10.1f}  {tag}")

    # 2. PCA ----------------------------------------------------------------
    hdr("2. PCA OF DAILY YIELD CHANGES (factor structure)")
    # PCA on the coupon curve (>=1y); T-bills are a separate money-market
    # segment whose Fed-driven dynamics distort the curvature factor.
    coupon = df[[c for c in df.columns if c >= 1.0]]
    pca = pca_curve(coupon, use_changes=True)
    evr = pca.explained_variance_ratio[:3] * 100
    print(
        f"  Explained variance:  level {evr[0]:.1f}%   "
        f"slope {evr[1]:.1f}%   curvature {evr[2]:.1f}%   "
        f"(sum {evr.sum():.1f}%)\n"
    )
    load = pca.summary(3)
    load.index = [tenor_label(t) for t in load.index]
    print(load.round(3).to_string())

    # 3. Carry & roll -------------------------------------------------------
    hdr("3. CARRY + ROLL-DOWN, 3M HORIZON (latest curve)")
    strat = ButterflyRVStrategy()
    cr = strat.carry_roll_snapshot(df, horizon_days=63)
    print("  expected carry+roll from holding each tenor 3 months:\n")
    for t, v in cr.items():
        bar = "#" * max(int(round(v / 2)), 0)
        print(f"  {tenor_label(t):>6} {v:>7.1f} bp  {bar}")

    # 4. Backtest -----------------------------------------------------------
    hdr("4. BUTTERFLY MEAN-REVERSION BACKTEST")
    flies = ", ".join(f.name for f in strat.flies)
    print(f"  Book: {flies}   (DV01-neutral, daily rebalance, 0.1bp cost)\n")
    res = strat.run(df, half_spread_bp=0.1)
    print(res.summary())

    # per-fly standalone Sharpe
    print("\n  Standalone Sharpe by fly:")
    for f in strat.flies:
        single = ButterflyRVStrategy(flies=[f]).run(df, half_spread_bp=0.1)
        print(f"    {f.name:>12} : {single.stats.sharpe:+.2f}")

    # 5. Cost sensitivity ---------------------------------------------------
    hdr("5. TRANSACTION-COST SENSITIVITY")
    print("  The edge is real but shallow -- costs dominate, the key RV lesson.\n")
    print(f"  {'half-spread(bp)':>16} {'Sharpe':>8} {'ann.PnL':>10}")
    for hs in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        r = strat.run(df, half_spread_bp=hs)
        print(f"  {hs:>16.2f} {r.stats.sharpe:>8.2f} {r.stats.ann_return:>10,.0f}")

    if make_charts:
        save_charts(df, fit, pca, res)


def save_charts(df, fit, pca, res) -> None:  # pragma: no cover - plotting
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[charts skipped: matplotlib not installed]")
        return

    DOCS.mkdir(exist_ok=True)

    # equity curve
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(res.equity.index, res.equity.values, lw=1.3, color="#1f77b4")
    ax.set_title("Butterfly mean-reversion: cumulative PnL")
    ax.set_ylabel("cumulative PnL ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "equity_curve.png", dpi=130)
    plt.close(fig)

    # NSS fit
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(fit.tenors.min(), fit.tenors.max(), 200)
    ax.plot(grid, fit.yield_at(grid), color="#333", label="NSS fit")
    ax.scatter(fit.tenors, fit.observed, color="#d62728", zorder=5, label="observed")
    ax.set_title(f"Treasury curve & NSS fit ({df.index[-1].date()})")
    ax.set_xlabel("tenor (years)")
    ax.set_ylabel("yield (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "nss_fit.png", dpi=130)
    plt.close(fig)

    # PCA loadings
    fig, ax = plt.subplots(figsize=(7, 4.5))
    load = pca.summary(3)
    for col in load.columns:
        ax.plot(load.index, load[col], marker="o", label=col)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("PCA loadings: level / slope / curvature")
    ax.set_xlabel("tenor (years)")
    ax.set_ylabel("loading")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "pca_loadings.png", dpi=130)
    plt.close(fig)

    print(f"\nCharts written to {DOCS}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="fetch fresh data from FRED")
    ap.add_argument("--charts", action="store_true", help="save PNG charts to docs/")
    args = ap.parse_args()

    df = fetch_fred() if args.live else load_sample()
    report(df, make_charts=args.charts)


if __name__ == "__main__":
    main()
