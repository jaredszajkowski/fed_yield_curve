"""
This module contains functions to load the datasets.
It is used to load the datasets into memory.

List of datasets:

- treas_yield_curve_zero_coupon: Federal Reserve yield curve

"""

import pull_fed_yield_curve

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

## treas_yield_curve_zero_coupon
df_all = pull_fed_yield_curve.load_fed_yield_curve(data_dir=DATA_DIR)
df = df_all.reset_index()
df = df.melt(id_vars=["Date"], var_name="tenor", value_name="y")
df = df[["tenor", "Date", "y"]].sort_values(by=["tenor", "Date"])
df = df.rename(columns={"Date": "ds", "tenor": "unique_id"})
df = df.dropna()
df.to_parquet(DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet")
