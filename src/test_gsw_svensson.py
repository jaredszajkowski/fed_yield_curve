"""Verify the pipeline output against Gurkaynak, Sack & Wright (2007).

Two complementary checks:

1. ``test_svensson_roundtrip`` (hermetic, no credentials) — the Fed dataset
   publishes *both* the fitted Svensson parameters (BETA0-3, TAU1-2) and the
   fitted zero-coupon yields (SVENY01-30). The paper's Eq. (§2) says the second
   is a deterministic function of the first. We recompute SVENY from the
   parameters and assert it matches the published column. This validates that
   our reading of the model and our column wiring are correct.

2. ``test_gsw_fit_quality`` (CRSP-gated, skipped when WRDS data is absent) — GSW
   fit the curve to outstanding coupon securities to within a few bp. We price
   the securities off the published curve and assert the actual-vs-predicted
   yield RMS is small, checking the pipeline against the paper's fit claim.

Reference:
    Gurkaynak, Sack & Wright, "The US Treasury yield curve: 1961 to the
    present," Journal of Monetary Economics 54, no. 8 (2007): 2291-2304.
"""

import numpy as np
import pandas as pd
import pytest

from pull_fed_yield_curve import DATA_DIR
from gsw_pricing import (
    actual_vs_predicted_yields,
    extract_params_from_fed_row,
    svensson_spot,
)

# Fitted parameters carry ~4 significant figures in the published CSV, so the
# reconstructed yields agree with the published ones only up to that rounding
# (empirically ~5e-5 pp). 1e-3 pp (0.1 bp) is a comfortable, non-flaky bound.
ROUNDTRIP_TOL_PP = 1e-3

# GSW report fitting errors of a few bp. Our security universe is unfiltered
# (unlike the paper, which drops bills, on-the-runs, callables, <3mo, etc.), so
# a handful of specials inflate the tail; we bound the RMS, not the max.
FIT_RMS_TOL_BP = 25.0

# Cross-section dates used for the GSW figure reproductions (Fig 2 / Fig 6).
GSW_DATES = ["2005-05-24", "2006-05-09"]

FED_ALL = DATA_DIR / "fed_yield_curve_all.parquet"
CRSP_INFO = DATA_DIR / "CRSP_TFZ_INFO.parquet"
CRSP_DAILY = DATA_DIR / "CRSP_TFZ_DAILY.parquet"


def _fed_all():
    if not FED_ALL.exists():
        pytest.skip(f"{FED_ALL} missing — run `doit pull:fed_yield_curve` first")
    return pd.read_parquet(FED_ALL)


def _param_rows(df, n=25):
    """The last ``n`` dates with a complete set of Svensson parameters."""
    need = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    return df.dropna(subset=need).tail(n)


def test_svensson_roundtrip():
    """Recomputed SVENY01-30 must match the published columns."""
    df = _fed_all()
    rows = _param_rows(df)
    assert not rows.empty, "no rows with a full parameter set"

    tenors = np.arange(1, 31)
    sveny_cols = [f"SVENY{n:02d}" for n in tenors]

    max_err = 0.0
    for _, row in rows.iterrows():
        params = extract_params_from_fed_row(row)
        recomputed = svensson_spot(tenors, params) * 100  # decimals -> percent
        published = row[sveny_cols].to_numpy(dtype=float)

        mask = ~np.isnan(published)
        assert mask.any(), f"no published SVENY on {row.name.date()}"
        err = np.abs(recomputed[mask] - published[mask])
        max_err = max(max_err, err.max())
        np.testing.assert_allclose(
            recomputed[mask],
            published[mask],
            atol=ROUNDTRIP_TOL_PP,
            err_msg=f"Svensson round-trip mismatch on {row.name.date()}",
        )

    assert max_err < ROUNDTRIP_TOL_PP


@pytest.mark.skipif(
    not (CRSP_DAILY.exists() and CRSP_INFO.exists()),
    reason="CRSP data absent — run `doit pull:crsp_treasury` (needs WRDS)",
)
@pytest.mark.parametrize("quote_date", GSW_DATES)
def test_gsw_fit_quality(quote_date):
    """Actual-vs-predicted yield RMS off the published curve must be small."""
    df = _fed_all()
    info = pd.read_parquet(CRSP_INFO)
    daily = pd.read_parquet(CRSP_DAILY)

    qd = pd.Timestamp(quote_date)
    assert qd in df.index, f"{quote_date} not in Fed dataset"

    res = actual_vs_predicted_yields(qd, daily, info, df.loc[qd])
    diff = (res["actual_yield"] - res["predicted_yield"]).dropna()
    assert len(diff) > 50, f"too few priced securities on {quote_date}: {len(diff)}"

    rms_bp = np.sqrt((diff**2).mean()) * 100
    assert rms_bp < FIT_RMS_TOL_BP, (
        f"{quote_date}: actual-vs-predicted RMS {rms_bp:.1f} bp "
        f"exceeds {FIT_RMS_TOL_BP} bp"
    )
