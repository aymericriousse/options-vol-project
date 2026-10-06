import numpy as np
import pandas as pd

from .implied_vol import implied_vol


def implied_forward(chain_exp, r, T):
    """Forward implicite d'une maturité par la parité call-put :
    C - P = e^(-rT)·(F - K)  =>  F = K + e^(rT)·(C - P).
    On utilise le strike où C et P sont les plus proches (près de la monnaie)."""
    calls = chain_exp[chain_exp["type"] == "call"].set_index("strike")["mid"]
    puts = chain_exp[chain_exp["type"] == "put"].set_index("strike")["mid"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return np.nan
    diff = calls[common] - puts[common]
    K_atm = diff.abs().idxmin()
    return K_atm + np.exp(r * T) * diff[K_atm]


def compute_smiles(chain, S0, r):
    """Vol implicite des options hors de la monnaie, pour chaque maturité."""
    rows = []
    for expiry, grp in chain.groupby("expiry"):
        T = grp["T"].iloc[0]
        F = implied_forward(grp, r, T)
        if np.isnan(F):
            continue
        q = r - np.log(F / S0) / T   # taux de dividende implicite : F = S·e^((r-q)T)

        otm = grp[((grp["type"] == "put") & (grp["strike"] < F)) |
                  ((grp["type"] == "call") & (grp["strike"] >= F))]

        for _, row in otm.iterrows():
            vol = implied_vol(row["mid"], S0, row["strike"], T, r, row["type"], q)
            rows.append({
                "expiry": expiry, "T": T, "strike": row["strike"], "type": row["type"],
                "mid": row["mid"], "F": F, "q": q,
                "k": np.log(row["strike"] / F),   # log-moneyness
                "iv": vol, "iv_yahoo": row["impliedVolatility"],
            })

        smiles = pd.DataFrame(rows).dropna(subset=["iv"])
    return smiles.sort_values(["expiry", "k"]).reset_index(drop=True)