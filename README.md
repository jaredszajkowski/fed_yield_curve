# Federal Reserve Yield Curve Pipeline

This pipeline downloads and processes Treasury yield curve data from the Federal Reserve.

## Data Source

The data is publicly available from the [Federal Reserve](https://www.federalreserve.gov/data/nominal-yield-curve.htm).

## Reference

Gurkaynak, Refet S., Brian Sack, and Jonathan H. Wright. "The US Treasury yield curve: 1961 to the present."
Journal of Monetary Economics 54.8 (2007): 2291-2304.

## Outputs

- `ftsfr_treas_yield_curve_zero_coupon.parquet`: Daily zero-coupon Treasury yields (1Y to 30Y)

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   doit
   ```

3. View the generated documentation in `docs/index.html`

## Yield Data

The dataset contains zero-coupon yields (SVENY01 to SVENY30) for maturities from 1 to 30 years,
estimated using the Gurkaynak, Sack, and Wright (2007) methodology.

## Academic References

### Primary Paper

- **Gurkaynak, Sack, and Wright (2007)** - "The U.S. Treasury yield curve: 1961 to the present"
  - Journal of Monetary Economics 54.8 (2007): 2291-2304
  - Federal Reserve official methodology for yield curve estimation

### Key Findings

- The Svensson (1994) parametric model provides a good fit to Treasury yields
- Zero-coupon yields can be reliably estimated from coupon bond prices
- The methodology is used by the Federal Reserve for official yield curve estimates
