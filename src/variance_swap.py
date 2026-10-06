import numpy as np
from scipy.stats import norm

from .svi import svi_total_variance


def otm_forward_price(k, w):
    """Prix d'une option hors de la monnaie, non actualisé et divisé par le forward (Black).
    Put si k < 0, call si k >= 0. w = variance totale implicite."""
    sw = np.sqrt(w)
    d1 = (-k + 0.5 * w) / sw
    d2 = d1 - sw
    call = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    put = np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1)
    return np.where(k < 0, put, call)


def var_swap_strike_svi(T, params, k_min=-30.0, k_max=5.0, n=200_001):
    """Strike de variance annualisé par réplication sur le smile SVI :
    K_var = (2/T) ∫ õ(k) e^(-k) dk."""
    k = np.linspace(k_min, k_max, n)
    w = np.maximum(svi_total_variance(k, *params), 1e-12)
    integrand = otm_forward_price(k, w) * np.exp(-k)
    return 2.0 / T * np.trapezoid(integrand, k)


def var_swap_strike_heston(T, v0, kappa, theta):
    """Strike de variance sous Heston (formule fermée) : variance moyenne espérée sur [0, T]."""
    return theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)