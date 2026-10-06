import numpy as np
from scipy.optimize import least_squares

from .black_scholes import bs_price, bs_vega


def heston_cf(u, T, v0, kappa, theta, xi, rho):
    """Fonction caractéristique de X_T = ln(S_T / F) sous Heston.
    Formulation stable dite "little Heston trap" (Albrecher et al.)."""
    iu = 1j * u
    beta = kappa - rho * xi * iu
    d = np.sqrt(beta ** 2 + xi ** 2 * (iu + u ** 2))
    g = (beta - d) / (beta + d)
    exp_dT = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * ((beta - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g)))
    D = ((beta - d) / xi ** 2) * (1 - exp_dT) / (1 - g * exp_dT)
    return np.exp(C + D * v0)


def heston_call(F, K, T, r, v0, kappa, theta, xi, rho, u_max=500.0, n=4001):
    """Prix de calls européens sous Heston (formule de Lewis).
    K peut être un tableau : tous les strikes d'une maturité en une fois."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    u = np.linspace(0.0, u_max, n)
    phi = heston_cf(u - 0.5j, T, v0, kappa, theta, xi, rho)         # ne dépend pas de K
    k = np.log(F / K)
    integrand = np.real(np.exp(1j * np.outer(k, u)) * phi) / (u ** 2 + 0.25)
    integral = np.trapezoid(integrand, u, axis=1)
    return np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)


def calibrate_heston(F, K, T, r, iv_mkt, x0=None):
    """Calibre Heston sur des vols implicites (un point par option).
    F, K, T, iv_mkt : tableaux 1D de même longueur.
    Erreur = (prix Heston - prix marché) / vega  ≈  erreur en vol."""
    F, K, T, iv_mkt = (np.asarray(x, dtype=float) for x in (F, K, T, iv_mkt))
    # Prix "marché" à partir des vols : Black-Scholes écrit sur le forward (S = F, q = r)
    price_mkt = bs_price(F, K, T, r, iv_mkt, "call", q=r)
    vega_mkt = bs_vega(F, K, T, r, iv_mkt, q=r)
    maturities = np.unique(T)

    def residuals(p):
        model = np.empty_like(price_mkt)
        for t in maturities:
            idx = T == t
            model[idx] = heston_call(F[idx][0], K[idx], t, r, *p)
        return (model - price_mkt) / vega_mkt

    #          v0     kappa  theta   xi     rho
    lower = [1e-4,  0.01,  1e-4,  0.01, -0.999]
    upper = [1.0,   20.0,  1.0,   5.0,   0.999]
    if x0 is None:
        x0 = [0.02, 2.0, 0.04, 1.0, -0.7]
    res = least_squares(residuals, x0, bounds=(lower, upper))
    return res.x, res