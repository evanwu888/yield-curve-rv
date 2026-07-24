"""Principal-component analysis of the yield curve.

Run on *daily yield changes* (in basis points) across tenors, the first three
principal components of the Treasury curve are famously interpretable:

    PC1  ~  level      (parallel shifts; ~90% of variance)
    PC2  ~  slope      (steepening / flattening)
    PC3  ~  curvature  (butterfly / belly moves)

These factors are the backbone of curve relative value: a DV01-neutral trade is
really a bet on one PC while hedging the others. We sign-normalise the loadings
so PC1 is a positive level move and PC2 is a bull-steepener direction, making
the output stable across samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    tenors: np.ndarray
    loadings: np.ndarray  # (n_tenors, n_components), columns are eigenvectors
    scores: pd.DataFrame  # (dates, n_components) factor time series
    explained_variance_ratio: np.ndarray
    mean: np.ndarray

    def summary(self, n: int = 3) -> pd.DataFrame:
        names = ["level", "slope", "curvature"] + [
            f"pc{i+1}" for i in range(3, self.loadings.shape[1])
        ]
        idx = names[: self.loadings.shape[1]]
        df = pd.DataFrame(self.loadings, index=self.tenors, columns=idx)
        return df.iloc[:, :n]


def pca_curve(yields: pd.DataFrame, use_changes: bool = True) -> PCAResult:
    """Principal components of the curve.

    Parameters
    ----------
    yields: DataFrame indexed by date, columns = tenor (years), values in %.
    use_changes: if True (default) run PCA on daily changes (bp); otherwise on
        yield levels. Changes are the standard choice for trading applications.
    """
    data = yields.sort_index()
    if use_changes:
        # daily changes in basis points
        mat = data.diff().dropna() * 100.0
    else:
        mat = data.dropna()

    X = mat.to_numpy(dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean

    # SVD is the numerically stable route to PCA.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    loadings = Vt.T  # (n_tenors, n_components)
    var = (S**2) / (len(Xc) - 1)
    evr = var / var.sum()

    # Sign convention: PC1 positive (level up), PC2 negative-at-short/positive-
    # at-long is a steepener -> make long-end loading positive.
    tenors = np.asarray(data.columns, dtype=float)
    if loadings[:, 0].sum() < 0:
        loadings[:, 0] *= -1
        U[:, 0] *= -1
    if loadings[:, 1].shape[0] and loadings[-1, 1] < 0:
        loadings[:, 1] *= -1
        U[:, 1] *= -1

    scores = pd.DataFrame(
        U * S,
        index=mat.index,
        columns=[f"pc{i+1}" for i in range(loadings.shape[1])],
    )
    return PCAResult(
        tenors=tenors,
        loadings=loadings,
        scores=scores,
        explained_variance_ratio=evr,
        mean=mean,
    )
