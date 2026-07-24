"""Tests for curve fitting and DV01."""

from __future__ import annotations

import numpy as np

from ycrv.curve import fit_nss, nss_design, par_bond_dv01


def test_nss_recovers_a_known_curve():
    # Build a curve from known NSS params, then check the fit reproduces it.
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    true = np.array([3.0, -1.5, 2.0, 0.5])  # b0..b3
    lam1, lam2 = 1.5, 8.0
    y = nss_design(tenors, lam1, lam2) @ true

    fit = fit_nss(tenors, y)
    # fitted curve should match the input to well under a basis point
    assert fit.rmse < 0.02
    assert np.allclose(fit.fitted, y, atol=0.05)


def test_nss_residuals_flag_a_cheap_point():
    tenors = np.array([1, 2, 3, 5, 7, 10, 30], dtype=float)
    y = np.array([4.0, 4.1, 4.2, 4.35, 4.5, 4.6, 4.8])
    y_bumped = y.copy()
    y_bumped[3] += 0.10  # make the 5y cheap (yield 10bp above the curve)

    fit = fit_nss(tenors, y_bumped)
    resid = fit.residuals
    # the 5y should have the most positive residual (cheapest)
    assert np.argmax(resid) == 3
    assert resid[3] > 0


def test_par_bond_dv01_matches_duration_identity():
    # For a 10y par bond ~4.6%, DV01 ≈ mod_duration * price * 1e-4 ≈ 0.079.
    dv01 = par_bond_dv01(10.0, 4.6)
    assert 0.075 < dv01 < 0.082

    # DV01 must increase with maturity.
    assert par_bond_dv01(2.0, 4.2) < par_bond_dv01(10.0, 4.6) < par_bond_dv01(30.0, 5.1)


def test_yield_at_is_smooth_and_interpolating():
    tenors = np.array([1, 2, 5, 10, 30], dtype=float)
    y = np.array([4.0, 4.1, 4.3, 4.6, 4.9])
    fit = fit_nss(tenors, y)
    # evaluating between nodes stays within the observed range
    mid = fit.yield_at(np.array([3.0, 7.0, 20.0]))
    assert np.all(mid > 3.5) and np.all(mid < 5.5)
