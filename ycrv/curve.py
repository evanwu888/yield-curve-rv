"""Yield-curve construction and DV01.

The Nelson-Siegel-Svensson (NSS) model represents the zero/par curve with six
parameters:

    y(tau) = b0
           + b1 * f1(tau; l1)
           + b2 * f2(tau; l1)
           + b3 * f3(tau; l2)

    f1 = (1 - exp(-tau/l1)) / (tau/l1)
    f2 = f1 - exp(-tau/l1)
    f3 = (1 - exp(-tau/l2)) / (tau/l2) - exp(-tau/l2)

The betas have clean economic readings: b0 is the long-run level, b1 the short
end (slope), b2/b3 the two humps (curvature). Crucially, for *fixed* decay
constants (l1, l2) the model is **linear** in the betas, so we fit betas with
ordinary least squares and only search the two decay parameters on a grid. That
keeps the fit fast, deterministic, and dependency-free (numpy only).

Fitting the observed curve gives, as a by-product, the residual of every tenor
versus the smooth fitted curve -- the raw "rich/cheap" signal that underlies
relative-value trades. A positive residual (yield above the curve) means the
bond is *cheap*; negative means *rich*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _phi1(tau: np.ndarray, lam: float) -> np.ndarray:
    x = tau / lam
    # limit as x -> 0 is 1; our tenors are strictly positive but guard anyway
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x == 0, 1.0, (1.0 - np.exp(-x)) / x)
    return out


def nss_design(tenors: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """Design matrix [1, f1, f2, f3] for the four NSS betas."""
    tau = np.asarray(tenors, dtype=float)
    f1 = _phi1(tau, lam1)
    f2 = f1 - np.exp(-tau / lam1)
    f3 = _phi1(tau, lam2) - np.exp(-tau / lam2)
    return np.column_stack([np.ones_like(tau), f1, f2, f3])


@dataclass
class NSSFit:
    betas: np.ndarray  # [b0, b1, b2, b3]
    lam1: float
    lam2: float
    tenors: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray
    rmse: float

    @property
    def residuals(self) -> np.ndarray:
        """observed - fitted, in the units of the yields (percent).

        Positive => cheap (yield above curve), negative => rich.
        """
        return self.observed - self.fitted

    def yield_at(self, tenor: float | np.ndarray) -> np.ndarray:
        """Evaluate the fitted curve at arbitrary tenor(s), in years."""
        design = nss_design(np.atleast_1d(tenor), self.lam1, self.lam2)
        return design @ self.betas

    def level_slope_curvature(self) -> dict[str, float]:
        """Readable NSS factor interpretation."""
        b0, b1, b2, b3 = self.betas
        return {
            "level": float(b0),
            "slope": float(b1),
            "curvature_1": float(b2),
            "curvature_2": float(b3),
        }


def fit_nss(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam1_grid: np.ndarray | None = None,
    lam2_grid: np.ndarray | None = None,
) -> NSSFit:
    """Fit NSS to a single observed curve via grid-search on (l1, l2) + OLS betas.

    Parameters
    ----------
    tenors: tenor of each yield, in years.
    yields: observed yields (percent); NaNs are dropped.
    lam1_grid, lam2_grid: candidate decay constants to search.
    """
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)
    mask = np.isfinite(tenors) & np.isfinite(yields)
    tenors, yields = tenors[mask], yields[mask]
    if len(tenors) < 4:
        raise ValueError("Need at least 4 valid points to fit NSS.")

    if lam1_grid is None:
        lam1_grid = np.linspace(0.4, 3.0, 27)
    if lam2_grid is None:
        lam2_grid = np.linspace(3.0, 14.0, 45)

    best: NSSFit | None = None
    for lam1 in lam1_grid:
        for lam2 in lam2_grid:
            # keep the two decay factors well separated so the curvature betas
            # stay identifiable (avoids the classic NSS lam1~=lam2 degeneracy)
            if lam2 <= lam1 * 1.5:
                continue
            design = nss_design(tenors, lam1, lam2)
            betas, *_ = np.linalg.lstsq(design, yields, rcond=None)
            fitted = design @ betas
            rmse = float(np.sqrt(np.mean((yields - fitted) ** 2)))
            if best is None or rmse < best.rmse:
                best = NSSFit(
                    betas=betas,
                    lam1=float(lam1),
                    lam2=float(lam2),
                    tenors=tenors,
                    observed=yields,
                    fitted=fitted,
                    rmse=rmse,
                )
    assert best is not None
    return best


def par_bond_dv01(tenor_years: float, ytm_pct: float, freq: int = 2) -> float:
    """DV01 of a par coupon bond: dollar price change per 1bp yield move.

    A bond trading at par ($100 face) has coupon = yield. We reprice the bond
    at +/-0.5bp and take the symmetric difference, which is the market-standard
    numerical DV01. Returned as a positive dollar amount per $100 face.
    """
    y = ytm_pct / 100.0
    n = max(int(round(tenor_years * freq)), 1)
    coupon = 100.0 * y / freq  # par bond

    def price(yld: float) -> float:
        r = yld / freq
        if r == 0:
            return coupon * n + 100.0
        disc = (1 + r) ** -np.arange(1, n + 1)
        return coupon * disc.sum() + 100.0 * disc[-1]

    bump = 1e-4  # 1bp
    return (price(y - bump) - price(y + bump)) / 2.0
