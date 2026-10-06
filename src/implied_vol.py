import numpy as np
from scipy.optimize import brentq

from .black_scholes import bs_price, bs_vega


def implied_vol(price, S, K, T, r, option_type="call", q=0.0, tol=1e-8, max_iter=100):
    """Volatilité implicite Black-Scholes.
    Newton-Raphson, avec repli sur la méthode de Brent si Newton échoue.
    Renvoie NaN si le prix viole les bornes d'arbitrage."""
    # --- 1. Bornes d'arbitrage ---
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if option_type == "call":
        lower, upper = max(S * disc_q - K * disc_r, 0.0), S * disc_q
    elif option_type == "put":
        lower, upper = max(K * disc_r - S * disc_q, 0.0), K * disc_r
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    if not (lower < price < upper):
        return np.nan

    # --- 2. Newton-Raphson ---
    sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type, q) - price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, T, r, sigma, q)
        if vega < 1e-8:          # vega trop petit : Newton devient instable
            break
        sigma -= diff / vega
        if sigma <= 0 or sigma > 5:   # Newton est parti hors des valeurs raisonnables
            break

    # --- 3. Repli : méthode de Brent (lente mais garantie) ---
    f = lambda s: bs_price(S, K, T, r, s, option_type, q) - price
    try:
        return brentq(f, 1e-6, 5.0, xtol=tol)
    except ValueError:
        return np.nan