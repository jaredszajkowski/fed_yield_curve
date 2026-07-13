"""Price CRSP Treasury securities off the GSW/Svensson curve.

Pure pricing helpers used by the summary notebook to reproduce GSW (2007)
Figures 2 and 6: for each coupon-bearing security trading on a given date, we
compute an *actual* yield (from its market price) and a *predicted* yield (from
the published Svensson parameters), both as bond-equivalent (semiannual) yields
so they are directly comparable and consistent with CRSP `tdyld`.

The Svensson pricing functions are adapted from the sibling `crsp_treasury`
pipeline's `calc_gsw_prices.py`.

Reference:
    Gurkaynak, Refet S., Brian Sack, and Jonathan H. Wright.
    "The US Treasury yield curve: 1961 to the present."
    Journal of Monetary Economics 54, no. 8 (2007): 2291-2304.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def svensson_spot(maturities, params):
    """Nelson-Siegel-Svensson continuously-compounded zero-coupon yield (decimals).

    params = (tau1, tau2, beta1, beta2, beta3, beta4) with betas in decimal form.
    """
    tau1, tau2, beta1, beta2, beta3, beta4 = params
    t = np.asarray(maturities, dtype=np.float64)
    tau1_term = (1 - np.exp(-t / tau1)) / (t / tau1)
    tau2_term = (1 - np.exp(-t / tau2)) / (t / tau2)
    return (
        beta1
        + beta2 * tau1_term
        + beta3 * (tau1_term - np.exp(-t / tau1))
        + beta4 * (tau2_term - np.exp(-t / tau2))
    )


def svensson_discount(maturities, params):
    """Discount factors d(t) = exp(-y(t) * t) from Svensson spot rates."""
    t = np.asarray(maturities, dtype=np.float64)
    return np.exp(-svensson_spot(t, params) * t)


def extract_params_from_fed_row(fed_row):
    """Convert a Fed params row (BETA0-3 in %, TAU1-2 in years) to model params.

    Returns (tau1, tau2, beta1, beta2, beta3, beta4) with betas in decimals.
    """
    return (
        fed_row["TAU1"],
        fed_row["TAU2"],
        fed_row["BETA0"] / 100,
        fed_row["BETA1"] / 100,
        fed_row["BETA2"] / 100,
        fed_row["BETA3"] / 100,
    )


def get_coupon_dates(quote_date, maturity_date):
    """Semiannual coupon dates strictly after quote_date through maturity_date."""
    quote_date = pd.to_datetime(quote_date)
    maturity_date = pd.to_datetime(maturity_date)
    n_periods = int(np.ceil((maturity_date - quote_date).days / 180))
    if n_periods <= 0:
        return None
    dates = pd.date_range(
        end=maturity_date, periods=n_periods, freq=pd.DateOffset(months=6)
    )
    dates = dates[dates > quote_date]
    return dates if len(dates) > 0 else None


def build_cashflows_for_date(df_date):
    """Cashflow matrix (rows = bonds, cols = payment dates) for one quote date.

    df_date must have columns caldt, tmatdt, tcouprt. Coupon = tcouprt/2 per $100
    face; principal = 100 at maturity.
    """
    quote_date = df_date["caldt"].iloc[0]
    all_dates = set()
    bond_info = {}
    for idx, row in df_date.iterrows():
        mat = row["tmatdt"]
        coupon_dates = get_coupon_dates(quote_date, mat)
        bond_info[idx] = (coupon_dates, row["tcouprt"], mat)
        if coupon_dates is not None:
            all_dates.update(coupon_dates)
        all_dates.add(mat)
    if not all_dates:
        return pd.DataFrame()
    all_dates = sorted(all_dates)
    CF = pd.DataFrame(0.0, index=df_date.index, columns=all_dates)
    for idx, (coupon_dates, coupon_rate, maturity) in bond_info.items():
        if coupon_dates is not None:
            for d in coupon_dates:
                if d in CF.columns:
                    CF.loc[idx, d] = coupon_rate / 2
        CF.loc[idx, maturity] += 100
    return CF.loc[:, (CF != 0).any()]


def bond_equiv_ytm(cashflows_vec, times, dirty_price):
    """Bond-equivalent (semiannual) YTM matching sum(CF/(1+y/2)^(2t)) = dirty_price."""
    if dirty_price is None or dirty_price <= 0 or len(cashflows_vec) == 0:
        return np.nan
    mask = cashflows_vec != 0
    cf, t = cashflows_vec[mask], times[mask]
    if len(cf) == 0:
        return np.nan

    def price_error(y):
        return np.sum(cf / (1 + y / 2) ** (2 * t)) - dirty_price

    for lo, hi in ((-0.05, 0.50), (-0.10, 1.00)):
        try:
            return brentq(price_error, lo, hi, xtol=1e-10, maxiter=200)
        except ValueError:
            continue
    return np.nan


def actual_vs_predicted_yields(quote_date, df_daily, df_info, fed_row):
    """Per-security actual and predicted yields for one date, for GSW Fig 2/6.

    Parameters
    ----------
    quote_date : pd.Timestamp
        The cross-section date.
    df_daily : pd.DataFrame
        CRSP daily quotes; needs kytreasno, caldt, and a dirty price. Accepts
        either a precomputed ``price_dirty`` column or ``tdbid``/``tdask``/
        ``tdaccint`` to build it.
    df_info : pd.DataFrame
        CRSP issue table; needs kytreasno, tmatdt, tcouprt, itype.
    fed_row : pd.Series
        Fed params row (BETA0-3, TAU1-2) for ``quote_date``.

    Returns
    -------
    pd.DataFrame
        Columns maturity_years, actual_yield, predicted_yield (yields in percent),
        one priceable coupon security per row.
    """
    day = df_daily[df_daily["caldt"] == quote_date].copy()
    if "price_dirty" not in day.columns:
        day["price_dirty"] = (
            (day["tdbid"] + day["tdask"]) / 2.0 + day["tdaccint"].fillna(0)
        )
    day = day.merge(
        df_info[["kytreasno", "tmatdt", "tcouprt", "itype"]], on="kytreasno", how="left"
    )

    # Priceable coupon-bearing notes/bonds with a valid quote and future maturity
    day = day[
        day["itype"].isin([1, 2])
        & (day["tcouprt"] > 0)
        & (day["tmatdt"] > quote_date)
        & (day["price_dirty"] > 0)
    ].reset_index(drop=True)
    if day.empty:
        return pd.DataFrame(
            columns=["maturity_years", "actual_yield", "predicted_yield"]
        )

    day["caldt"] = quote_date
    params = extract_params_from_fed_row(fed_row)

    CF = build_cashflows_for_date(day)
    times = np.array([(d - quote_date).days / 365.25 for d in CF.columns])
    times = np.maximum(times, 1e-6)
    disc = svensson_discount(times, params)
    predicted_dirty = CF.values @ disc  # model-implied dirty price per bond

    out = []
    cf_values = CF.values
    for i, idx in enumerate(CF.index):
        cf = cf_values[i]
        actual = bond_equiv_ytm(cf, times, day.loc[idx, "price_dirty"])
        predicted = bond_equiv_ytm(cf, times, predicted_dirty[i])
        out.append(
            {
                "maturity_years": (day.loc[idx, "tmatdt"] - quote_date).days / 365.25,
                "actual_yield": actual * 100,
                "predicted_yield": predicted * 100,
            }
        )
    return pd.DataFrame(out)
