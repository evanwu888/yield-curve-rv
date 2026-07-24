"""Treasury yield-curve data loading.

Two sources are supported:

* ``load_sample()`` reads the real (bundled) daily U.S. Treasury par-yield
  history shipped with the repo, so everything runs offline out of the box.
* ``fetch_fred()`` pulls fresh data live from the St. Louis Fed (FRED). It uses
  the public CSV download endpoint and needs **no API key**.

The canonical in-memory representation used everywhere else in the library is a
``pandas.DataFrame`` indexed by date, with one column per tenor. Column labels
are tenors expressed in **years** (floats), e.g. ``0.0833`` for 1-month,
``2.0`` for 2-year. Values are yields in **percent** (so ``4.25`` means 4.25%).
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

# FRED constant-maturity Treasury series -> tenor in years.
# https://fred.stlouisfed.org/categories/115
FRED_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SAMPLE_CSV = _DATA_DIR / "ust_yields.csv"


def load_sample() -> pd.DataFrame:
    """Load the bundled daily Treasury yield history (offline, real data)."""
    if not _SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Bundled dataset not found at {_SAMPLE_CSV}. "
            "Regenerate it with `python -m ycrv.data`."
        )
    df = pd.read_csv(_SAMPLE_CSV, index_col=0, parse_dates=True)
    df.columns = [float(c) for c in df.columns]
    return df.sort_index()


def _fetch_one(series_id: str, start: str, end: str | None) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={start}"
    )
    if end:
        url += f"&coed={end}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
        raw = resp.read().decode("utf-8")
    s = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True).iloc[:, 0]
    # FRED marks missing observations with ".".
    return pd.to_numeric(s, errors="coerce")


def fetch_fred(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch constant-maturity Treasury yields live from FRED.

    Parameters
    ----------
    start, end:
        ISO date strings (``end=None`` means "up to today").

    Returns
    -------
    DataFrame indexed by date, columns = tenor in years, values = yield in %.
    """
    cols: dict[float, pd.Series] = {}
    for series_id, tenor in FRED_SERIES.items():
        cols[tenor] = _fetch_one(series_id, start, end)
    df = pd.DataFrame(cols).sort_index(axis=1).sort_index()
    # Keep business days on which at least the core curve is observed.
    df = df.dropna(how="all")
    return df


def save_sample(df: pd.DataFrame) -> Path:
    """Persist a yield DataFrame as the bundled sample dataset."""
    _DATA_DIR.mkdir(exist_ok=True)
    out = df.copy()
    out.columns = [f"{c:.6g}" for c in out.columns]
    out.to_csv(_SAMPLE_CSV)
    return _SAMPLE_CSV


if __name__ == "__main__":
    # Regenerate the bundled dataset from live FRED data.
    print("Fetching Treasury yields from FRED ...")
    data = fetch_fred(start="2010-01-01")
    path = save_sample(data)
    print(f"Saved {len(data):,} rows x {data.shape[1]} tenors -> {path}")
