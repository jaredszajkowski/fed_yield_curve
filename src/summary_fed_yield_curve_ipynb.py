# %%
"""
# Federal Reserve Treasury Yield Curve

This notebook provides summary statistics and visualizations for the Federal Reserve's
Treasury yield curve data based on the Gurkaynak, Sack, and Wright (2007) model.

## Data Source

Data is publicly available from the [Federal Reserve](https://www.federalreserve.gov/data/nominal-yield-curve.htm).

## Reference

Gurkaynak, Refet S., Brian Sack, and Jonathan H. Wright. "The US Treasury yield curve: 1961 to the present."
Journal of Monetary Economics 54.8 (2007): 2291-2304.
"""

# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

warnings.filterwarnings("ignore")

# %%
"""
## Load the Dataset

The dataset contains zero-coupon Treasury yields for maturities from 1 to 30 years.
"""

# %%
# Load the yield curve data
df = pd.read_parquet(DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet")
print(f"Dataset shape: {df.shape}")
df.head()

# %%
# Pivot to wide format for analysis
df_wide = df.pivot(index='ds', columns='unique_id', values='y')
print(f"Wide format shape: {df_wide.shape}")
print(f"Tenors: {df_wide.columns.tolist()}")

# %%
"""
## Summary Statistics
"""

# %%
# Summary statistics for each tenor
summary_stats = df_wide.describe().T
summary_stats

# %%
"""
## Data Coverage
"""

# %%
print(f"Date range: {df_wide.index.min()} to {df_wide.index.max()}")
print(f"Number of observations: {len(df_wide)}")
print(f"Missing values per tenor:")
print(df_wide.isnull().sum())

# %%
"""
## Current Yield Curve

The most recent yield curve snapshot.
"""

# %%
# Get the most recent yield curve
latest_date = df_wide.index.max()
latest_curve = df_wide.loc[latest_date].dropna()

# Extract maturity in years from tenor names (e.g., SVENY01 -> 1)
maturities = [int(col.replace('SVENY', '')) for col in latest_curve.index]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(maturities, latest_curve.values, 'b-o', linewidth=2, markersize=6)
ax.set_title(f'Treasury Yield Curve as of {latest_date.strftime("%Y-%m-%d")}', fontsize=14)
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Yield (%)')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 31, 5))
plt.tight_layout()
plt.show()

# %%
"""
## Historical Yield Curve Evolution

Time series of selected tenors.
"""

# %%
# Select key tenors for visualization
key_tenors = ['SVENY01', 'SVENY02', 'SVENY05', 'SVENY10', 'SVENY30']
available_tenors = [t for t in key_tenors if t in df_wide.columns]

fig, ax = plt.subplots(figsize=(14, 6))
for tenor in available_tenors:
    label = tenor.replace('SVENY', '') + 'Y'
    ax.plot(df_wide.index, df_wide[tenor], label=label, linewidth=0.8)

ax.set_title('Historical Treasury Yields by Maturity', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Yield (%)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
"""
## Yield Curve Slope (10Y - 2Y Spread)
"""

# %%
if 'SVENY10' in df_wide.columns and 'SVENY02' in df_wide.columns:
    spread_10_2 = df_wide['SVENY10'] - df_wide['SVENY02']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(spread_10_2.index, spread_10_2.values, linewidth=0.8, color='green')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.fill_between(spread_10_2.index, 0, spread_10_2.values,
                    where=(spread_10_2 < 0), color='red', alpha=0.3, label='Inverted')
    ax.fill_between(spread_10_2.index, 0, spread_10_2.values,
                    where=(spread_10_2 >= 0), color='green', alpha=0.3, label='Normal')
    ax.set_title('Treasury 10Y-2Y Spread (Yield Curve Slope)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nSpread Statistics:")
    print(f"  Mean: {spread_10_2.mean():.2f}%")
    print(f"  Std: {spread_10_2.std():.2f}%")
    print(f"  Min: {spread_10_2.min():.2f}%")
    print(f"  Max: {spread_10_2.max():.2f}%")
    print(f"  Current: {spread_10_2.iloc[-1]:.2f}%")

# %%
"""
## Correlation Matrix

Correlation between yields at different maturities.
"""

# %%
# Select subset of tenors for correlation
corr_tenors = ['SVENY01', 'SVENY02', 'SVENY05', 'SVENY07', 'SVENY10', 'SVENY20', 'SVENY30']
available_corr = [t for t in corr_tenors if t in df_wide.columns]
corr_matrix = df_wide[available_corr].corr()

# Rename for display
rename_dict = {t: t.replace('SVENY', '') + 'Y' for t in available_corr}
corr_matrix = corr_matrix.rename(index=rename_dict, columns=rename_dict)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0.9,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Matrix: Treasury Yields Across Maturities', fontsize=12)
plt.tight_layout()
plt.show()

# %%
"""
## Summary

This dataset provides daily zero-coupon Treasury yields estimated using the
Gurkaynak, Sack, and Wright (2007) model. The yields cover maturities from 1 to 30 years
and are widely used in fixed income research and practice.

Key applications:
- Yield curve analysis and term structure modeling
- Fixed income derivatives pricing
- Monetary policy analysis
- Risk-free rate estimation
"""
