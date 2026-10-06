# Options Pricing & Implied Volatility Surface — SPY

Python project covering the full chain from option pricing models to market-implied volatility:
three independent pricing methods validated against each other, a custom implied-volatility solver,
and the implied volatility surface of SPY options built from real market data.

![SPY implied volatility surface](figures/vol_surface_spy.png)

## Contents

| Step | Topic | Status |
|---|---|---|
| 1 | European option pricing: closed form, Monte Carlo, finite differences (Crank-Nicolson) | ✅ |
| 2 | SPY market data, implied volatility solver, smiles and volatility surface | ✅ |
| 3 | Model calibration (SVI, Heston) and exotic option pricing | 🔄 |
| 4 | Delta-hedging backtest: realised vs implied volatility, transaction costs | ⏳ |

---

## Step 1 — Pricing a European call three ways

Test case: S = 100, K = 100, T = 1, r = 5%, σ = 20%.

| Method | Price | Comment |
|---|---|---|
| Black-Scholes closed form | 10.4506 | Reference |
| Monte Carlo (1M paths) | 10.4532 ± 0.0289 (95% CI) | Error decreases as 1/√N |
| Crank-Nicolson PDE (400×400 grid) | 10.4481 | Second-order convergence |
| Crank-Nicolson + Richardson extrapolation | 10.45059 | Error ≈ 4·10⁻⁶ |

![Monte Carlo convergence](figures/mc_convergence.png)

**Key takeaways**
- Monte Carlo converges slowly (1/√N): 100× more paths only divides the error by 10.
- The Crank-Nicolson scheme is confirmed to be second order: doubling the grid divides the error by 4.00.
- Richardson extrapolation, combining the 200 and 400 grids, cancels the leading error term
  and improves accuracy by roughly three orders of magnitude.
- In one dimension, finite differences are far more efficient than Monte Carlo;
  Monte Carlo becomes the method of choice for path-dependent or multi-asset products.

Greeks (delta, gamma, vega) are implemented analytically, with a continuous dividend yield.

---

## Step 2 — SPY implied volatility surface

**Data:** full SPY option chain downloaded with `yfinance` (snapshot of 2026-10-06, spot = 779.56),
risk-free rate from the 13-week T-bill. 10,014 quotes, 8,082 after cleaning
(positive bid, maturity ≥ 7 days, relative bid-ask spread ≤ 50%), 26 maturities from 7 to 836 days.

**Methodology**
- **Implied forward from put-call parity** for each maturity, F = K + e^(rT)(C − P),
  which gives the market-implied dividend yield instead of assuming one.
- **Out-of-the-money options only** (puts below the forward, calls above): they are the most liquid,
  and their early-exercise premium is negligible, which limits the bias of applying
  European Black-Scholes to American SPY options.
- **Custom implied volatility solver:** Newton-Raphson using the analytical vega,
  with a fallback to Brent's method, and no-arbitrage bounds checked beforehand.

![SPY smiles by maturity](figures/smiles_spy.png)

**Observations**
- **Pronounced skew:** at 17 days, implied vol goes from ~60% (k = −0.4) to ~10% at the money.
  It reflects crash risk, demand for protective puts and the negative spot-volatility correlation.
- **The skew flattens with maturity**, and **at-the-money vol increases with maturity**
  (upward-sloping term structure, typical of a calm market).
- **The options "see" the dividend date:** the implied dividend yield is close to zero
  for maturities before SPY's December ex-dividend date and becomes positive afterwards.
- **Consistency check:** the smile obtained is continuous across the put/call switch at the forward,
  whereas the implied volatilities provided by Yahoo Finance show a visible jump at the money.

**Limitation:** the raw surface shows some noise near the money at short maturities
(bid-ask spreads, asynchronous quotes). This motivates step 3: fitting a parametric,
arbitrage-free model to the surface.

---

## Project structure

```
├── data/          # Market data snapshots (CSV)
├── figures/       # Charts used in this README
├── notebooks/
│   ├── 01_pricer_validation.ipynb   # Step 1: pricing methods and convergence
│   └── 02_market_data.ipynb         # Step 2: market data, implied vol, surface
└── src/
    ├── black_scholes.py       # Closed-form prices and Greeks
    ├── monte_carlo.py         # Monte Carlo pricer
    ├── finite_differences.py  # Crank-Nicolson PDE solver
    ├── implied_vol.py         # Implied volatility solver
    ├── data.py                # Market data download and cleaning
    └── vol_surface.py         # Implied forward and smiles
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Then open the notebooks in `notebooks/` and run all cells.

---

*Aymeric Riousse — MSc Quantitative Finance, ECE Paris ·
[LinkedIn](https://www.linkedin.com/in/aymericriousse/)*