import numpy as np

from .black_scholes import bs_price, bs_delta, bs_gamma


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    """Trajectoires de mouvement brownien géométrique. Renvoie un tableau (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def delta_hedge_pnl(paths, K, T, r, sigma_sold, sigma_hedge=None, option_type="call", cost=0.0):
    """P&L à maturité d'un trader qui VEND une option et la couvre en delta.
    - vend l'option au prix Black-Scholes avec la vol sigma_sold (la vol implicite)
    - couvre avec le delta calculé à la vol sigma_hedge (par défaut = sigma_sold)
    - réajuste la couverture à chaque pas de la trajectoire
    - cost : coût de transaction proportionnel au montant échangé (0.0005 = 5 points de base)"""
    if sigma_hedge is None:
        sigma_hedge = sigma_sold
    n = paths.shape[1] - 1
    dt = T / n

    # t = 0 : on encaisse la prime et on achète delta actions
    S = paths[:, 0]
    premium = bs_price(S, K, T, r, sigma_sold, option_type)
    delta = bs_delta(S, K, T, r, sigma_hedge, option_type)
    cash = premium - delta * S - cost * np.abs(delta) * S

    # Réajustements intermédiaires
    for i in range(1, n):
        S = paths[:, i]
        cash = cash * np.exp(r * dt)                      # le cash rapporte (ou coûte) des intérêts
        new_delta = bs_delta(S, K, T - i * dt, r, sigma_hedge, option_type)
        trade = new_delta - delta
        cash -= trade * S + cost * np.abs(trade) * S
        delta = new_delta

    # Maturité : on revend les actions et on paie le payoff de l'option
    S_T = paths[:, -1]
    cash = cash * np.exp(r * dt)
    cash += delta * S_T - cost * np.abs(delta) * S_T
    payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    return cash - payoff


def gamma_pnl_approx(paths, K, T, r, sigma, option_type="call"):
    """P&L théorique du vendeur couvert : somme de ½·Γ·S²·(σ²·dt − rendement²)."""
    n = paths.shape[1] - 1
    dt = T / n
    total = np.zeros(paths.shape[0])
    for i in range(n):
        S = paths[:, i]
        gamma = bs_gamma(S, K, T - i * dt, r, sigma)
        ret2 = (paths[:, i + 1] / S - 1) ** 2
        total += 0.5 * gamma * S ** 2 * (sigma ** 2 * dt - ret2) * np.exp(r * (T - (i + 1) * dt))
    return total