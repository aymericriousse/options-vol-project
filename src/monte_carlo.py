import numpy as np


def mc_price(S, K, T, r, sigma, option_type="call", q=0.0, n_paths=100_000, seed=None):
    """Prix d'une option européenne par Monte Carlo sous Black-Scholes.
    Retourne (prix, erreur standard)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    # Prix final de l'action : solution exacte du mouvement brownien géométrique
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)
    return price, std_error