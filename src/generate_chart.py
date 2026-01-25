"""Generate interactive HTML chart for Fed Yield Curve."""

import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# Get the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "_data"
OUTPUT_DIR = PROJECT_ROOT / "_output"


def generate_yield_curve_chart():
    """Generate Treasury yield curve time series chart."""
    # Load Treasury yield curve data
    df = pd.read_parquet(DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet")

    # Create line chart
    fig = px.line(
        df.sort_values("ds"),
        x="ds",
        y="y",
        color="unique_id",
        title="Treasury Zero-Coupon Yield Curve",
        labels={
            "ds": "Date",
            "y": "Yield (%)",
            "unique_id": "Maturity"
        }
    )

    # Update layout
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save chart
    output_path = OUTPUT_DIR / "yield_curve_replication.html"
    fig.write_html(str(output_path))
    print(f"Chart saved to {output_path}")

    return fig


if __name__ == "__main__":
    generate_yield_curve_chart()
