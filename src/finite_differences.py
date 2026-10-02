import numpy as np
from scipy.linalg import solve_banded


def fd_price(S, K, T, r, sigma, option_type="call", q=0.0, M=200, N=200, S_max_mult=4):
    """Prix d'une option européenne par différences finies (Crank-Nicolson)
    sur l'EDP de Black-Scholes.
    M : nombre de pas en spot, N : nombre de pas en temps."""
    # --- Grille ---
    S_max = S_max_mult * max(S, K)
    dS = S_max / M
    dt = T / N
    S_grid = np.linspace(0, S_max, M + 1)
    i = np.arange(1, M)  # indices des points intérieurs de la grille

    # --- Condition terminale : à maturité, le prix = le payoff ---
    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    # --- Coefficients de Crank-Nicolson ---
    a = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
    b = -0.5 * dt * (sigma**2 * i**2 + r)
    c = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)

    # Matrice tridiagonale du côté implicite (format "bande" pour scipy)
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = -c[:-1]   # diagonale supérieure
    ab[1, :] = 1 - b      # diagonale principale
    ab[2, :-1] = -a[1:]   # diagonale inférieure

    # --- On remonte le temps, de la maturité vers aujourd'hui ---
    for n in range(N):
        tau = (n + 1) * dt  # temps restant jusqu'à maturité

        # Conditions aux bords (S = 0 et S = S_max)
        if option_type == "call":
            V0, VM = 0.0, S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
        else:
            V0, VM = K * np.exp(-r * tau), 0.0

        # Côté explicite (connu) du schéma
        rhs = a * V[:-2] + (1 + b) * V[1:-1] + c * V[2:]
        rhs[0] += a[0] * V0
        rhs[-1] += c[-1] * VM

        # Côté implicite : on résout le système tridiagonal
        V[1:-1] = solve_banded((1, 1), ab, rhs)
        V[0], V[-1] = V0, VM

    # Prix au spot S, par interpolation sur la grille
    return np.interp(S, S_grid, V)