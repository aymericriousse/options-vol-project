import numpy as np
import pandas as pd

from .hedging import delta_hedge_pnl


def monthly_vol_selling(hist, horizon=21, cost=0.0, straddle_vol_shift=0.0):
    """Chaque mois (21 jours de bourse), vend la vol à 1 mois au niveau du VIX :
    - un variance swap (nominal vega de 1 $) : P&L en points de vol
    - un straddle ATM couvert en delta chaque jour : P&L en % du spot
    Fenêtres successives, sans chevauchement."""
    closes = hist["SPY"].values
    rows = []
    for start in range(0, len(hist) - horizon, horizon):
        path = closes[start:start + horizon + 1]
        S0 = path[0]
        vix = hist["VIX"].iloc[start] / 100
        r = hist["IRX"].iloc[start] / 100
        T = horizon / 252

        # Vol réalisée sur le mois suivant
        rets = np.diff(np.log(path))
        realized = np.sqrt(252 / horizon * np.sum(rets ** 2))

        # Straddle ATM vendu au VIX et couvert chaque jour (une seule trajectoire : la vraie)
        p = path[None, :]
        straddle = (delta_hedge_pnl(p, S0, T, r, vix - straddle_vol_shift, option_type="call", cost=cost)
                    + delta_hedge_pnl(p, S0, T, r, vix - straddle_vol_shift, option_type="put", cost=cost))[0]

        rows.append({
            "date": hist.index[start],
            "spot": S0,
            "vix": vix,
            "realized": realized,
            "varswap_pnl": 100 * (vix ** 2 - realized ** 2) / (2 * vix),   # points de vol
            "straddle_pnl_pct": 100 * straddle / S0,                        # % du spot
        })
    return pd.DataFrame(rows).set_index("date")