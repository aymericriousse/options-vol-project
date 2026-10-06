import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def svi_total_variance(k, a, b, rho, m, sigma):
    """SVI 'raw' (Gatheral) : variance totale implicite w(k) = σ_imp² · T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k, T, params):
    """Vol implicite donnée par SVI : σ_imp = sqrt(w / T)."""
    w = svi_total_variance(k, *params)
    return np.sqrt(np.maximum(w, 1e-12) / T)


def fit_svi_slice(k, iv, T, n_starts=10, seed=0):
    """Calibre SVI sur un smile (une maturité).
    - erreur mesurée en vol implicite (chaque option compte autant)
    - perte robuste soft_l1 (limite l'effet des points aberrants)
    - a peut être négatif, mais la variance minimale doit rester >= 0"""
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    w_mkt = iv ** 2 * T

    def residuals(p):
        a, b, rho, m, sigma = p
        iv_model = svi_implied_vol(k, T, p)
        w_min = a + b * sigma * np.sqrt(1 - rho ** 2)   # variance minimale de SVI
        penalty = 100.0 * max(0.0, -w_min)              # pénalité si elle devient négative
        return np.append(iv_model - iv, penalty)

    #          a              b      rho     m     sigma
    lower = [-w_mkt.max(),  1e-6, -0.999, -1.0, 1e-4]
    upper = [ w_mkt.max(), 10.0,   0.999,  1.0, 2.0]

    rng = np.random.default_rng(seed)
    best = None
    for i in range(n_starts):
        if i == 0:
            x0 = [0.5 * w_mkt.min(), 0.1, -0.5, 0.0, 0.1]
        else:
            x0 = [rng.uniform(-0.5, 1.0) * w_mkt.min(), rng.uniform(0.01, 1.0),
                  rng.uniform(-0.9, 0.5), rng.uniform(-0.2, 0.2), rng.uniform(0.01, 0.5)]
        res = least_squares(residuals, x0, bounds=(lower, upper),
                            loss="soft_l1", f_scale=0.01)
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def fit_svi_surface(smiles, min_points=8):
    """Calibre SVI maturité par maturité. Renvoie un tableau de paramètres."""
    rows = []
    for expiry, s in smiles.groupby("expiry"):
        if len(s) < min_points:
            continue
        T = s["T"].iloc[0]
        params = fit_svi_slice(s["k"].values, s["iv"].values, T)
        iv_fit = svi_implied_vol(s["k"].values, T, params)
        rmse = np.sqrt(np.mean((iv_fit - s["iv"].values) ** 2))
        a, b, rho, m, sigma = params
        rows.append({"expiry": expiry, "T": T, "a": a, "b": b, "rho": rho,
                     "m": m, "sigma": sigma, "rmse_vol_pts": rmse * 100, "n_points": len(s)})
    return pd.DataFrame(rows)