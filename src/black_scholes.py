import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Calcule d1 et d2 de Black-Scholes.
    S: spot, K: strike, T: maturité en années, r: taux sans risque,
    sigma: volatilité, q: taux de dividende continu
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Prix Black-Scholes d'un call ou d'un put européen."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Delta Black-Scholes : sensibilité du prix au spot."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """Gamma Black-Scholes : sensibilité du delta au spot (identique call/put)."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega Black-Scholes : sensibilité du prix à la vol (identique call/put).
    Exprimé pour une variation de 1 (= 100 points de vol)."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)