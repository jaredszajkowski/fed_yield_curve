from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"


def pull_fed_yield_curve():
    """
    Download the latest yield curve from the Federal Reserve

    This is the published data using Gurkaynak, Sack, and Wright (2007) model
    """

    url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    response = requests.get(url)
    pdf_stream = BytesIO(response.content)
    df_all = pd.read_csv(pdf_stream, skiprows=9, index_col=0, parse_dates=True)

    cols = ["SVENY" + str(i).zfill(2) for i in range(1, 31)]
    df = df_all[cols]
    return df_all, df


def load_fed_yield_curve_all(data_dir=DATA_DIR):
    path = data_dir / "fed_yield_curve_all.parquet"
    _df = pd.read_parquet(path)
    return _df


def load_fed_yield_curve(data_dir=DATA_DIR):
    path = data_dir / "fed_yield_curve.parquet"
    _df = pd.read_parquet(path)
    return _df


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_all, df = pull_fed_yield_curve()
    path = Path(DATA_DIR) / "fed_yield_curve_all.parquet"
    df_all.to_parquet(path)
    path = Path(DATA_DIR) / "fed_yield_curve.parquet"
    df.to_parquet(path)
