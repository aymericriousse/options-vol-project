import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import black_scholes as bs
from src.monte_carlo import mc_price
from src.finite_differences import fd_price
import src.config as config
import src.svi as svi
import src.heston as heston
from src.implied_vol import implied_vol
import src.variance_swap as vs
import src.hedging as hd
import src.data as data
import src.backtest as backtest
from plotly.subplots import make_subplots

st.set_page_config(page_title="SPY Volatility Lab", page_icon="📈", layout="wide")
st.title("SPY Volatility Lab")
st.caption("Pricing d'options, surface de volatilité et trading de volatilité sur SPY · Aymeric Riousse")

tab_pricer, tab_surface, tab_varswap, tab_hedge, tab_backtest = st.tabs(
    ["Pricer", "Surface de vol", "Variance swap", "Delta-hedging", "Backtest"])

# =============================================================
# Onglet 1 : Pricer
# =============================================================
with tab_pricer:
    st.subheader("Pricer d'options européennes")

    c1, c2, c3 = st.columns(3)
    S = c1.number_input("Spot S", value=100.0, min_value=1.0)
    K = c1.number_input("Strike K", value=100.0, min_value=1.0)
    T = c2.number_input("Maturité T (années)", value=1.0, min_value=0.01, step=0.25)
    r = c2.number_input("Taux r (%)", value=5.0, step=0.25) / 100
    sigma = c3.number_input("Volatilité σ (%)", value=20.0, min_value=1.0, step=1.0) / 100
    q = c3.number_input("Dividende q (%)", value=0.0, step=0.25) / 100
    opt = st.radio("Type d'option", ["call", "put"], horizontal=True)

    price = bs.bs_price(S, K, T, r, sigma, opt, q)
    m = st.columns(4)
    m[0].metric("Prix Black-Scholes", f"{price:.4f}")
    m[1].metric("Delta", f"{bs.bs_delta(S, K, T, r, sigma, opt, q):.4f}")
    m[2].metric("Gamma", f"{bs.bs_gamma(S, K, T, r, sigma, q):.4f}")
    m[3].metric("Vega (par point de vol)", f"{bs.bs_vega(S, K, T, r, sigma, q) / 100:.4f}")

    with st.expander("Comparer avec Monte Carlo et les différences finies"):
        mc, se = mc_price(S, K, T, r, sigma, opt, q, n_paths=200_000, seed=42)
        fd = fd_price(S, K, T, r, sigma, opt, q, M=400, N=400)
        st.dataframe(pd.DataFrame({
            "Méthode": ["Formule fermée", "Monte Carlo (200 000 trajectoires)", "Différences finies (400 × 400)"],
            "Prix": [price, mc, fd],
            "Écart avec la formule": [0.0, mc - price, fd - price],
        }).round(4), hide_index=True)

    # Prix de l'option en fonction du spot
    S_grid = np.linspace(0.5 * K, 1.5 * K, 200)
    payoff = np.maximum(S_grid - K, 0) if opt == "call" else np.maximum(K - S_grid, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_grid, y=bs.bs_price(S_grid, K, T, r, sigma, opt, q), name="Prix aujourd'hui"))
    fig.add_trace(go.Scatter(x=S_grid, y=payoff, name="Payoff à maturité", line=dict(dash="dash")))
    fig.update_layout(title="Prix de l'option en fonction du spot",
                      xaxis_title="Spot", yaxis_title="Prix", height=420)
    st.plotly_chart(fig)

# =============================================================
# Chargement des résultats de la photo du marché (une seule fois)
# =============================================================
@st.cache_data
def load_snapshot_results(date):
    smiles = pd.read_csv(config.DATA_DIR / f"spy_smiles_{date}.csv", parse_dates=["expiry"])
    svi_params = pd.read_csv(config.DATA_DIR / f"svi_params_{date}.csv", parse_dates=["expiry"])
    hp = pd.read_csv(config.DATA_DIR / f"heston_params_{date}.csv", index_col=0).iloc[:, 0].astype(float)
    market = pd.read_csv(config.DATA_DIR / f"spy_market_{date}.csv", index_col=0).iloc[:, 0]
    return smiles, svi_params, hp, float(market["spot"]), float(market["r"])


# =============================================================
# Onglet 2 : Surface de vol
# =============================================================
with tab_surface:
    date = config.SNAPSHOT_DATE
    smiles, svi_params, hp, S0, r_mkt = load_snapshot_results(date)
    st.subheader(f"Surface de volatilité implicite SPY – photo du {date}")
    st.caption(f"Spot = {S0:.2f} · taux sans risque = {r_mkt:.2%} · "
               f"{len(smiles)} options hors de la monnaie · {len(svi_params)} maturités")

    # Surface SVI, affichée uniquement sur la plage de strikes cotés de chaque maturité
    k_grid = np.linspace(-0.3, 0.3, 61)
    k_range = smiles.groupby("expiry")["k"].agg(["min", "max"])
    Z = []
    for _, row in svi_params.iterrows():
        vols = svi.svi_implied_vol(k_grid, row["T"], row[["a", "b", "rho", "m", "sigma"]].values.astype(float)) * 100
        lo, hi = k_range.loc[row["expiry"]]
        vols[(k_grid < lo) | (k_grid > hi)] = np.nan
        Z.append(vols)

    fig3d = go.Figure(go.Surface(x=k_grid, y=svi_params["T"] * 365, z=np.array(Z),
                                 colorscale="Viridis", colorbar=dict(title="Vol (%)")))
    if st.checkbox("Afficher les points de marché"):
        pts = smiles[smiles["k"].abs() <= 0.3]
        fig3d.add_trace(go.Scatter3d(x=pts["k"], y=pts["T"] * 365, z=pts["iv"] * 100, mode="markers",
                                     marker=dict(size=2, color="black"), name="Marché"))
    fig3d.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0),
                        scene=dict(xaxis_title="Log-moneyness k", yaxis_title="Maturité (jours)",
                                   zaxis_title="Vol implicite (%)"))
    st.plotly_chart(fig3d, config={"scrollZoom": False})

    # Smile d'une maturité choisie
    st.markdown("#### Smile d'une maturité")
    labels = [f"{e.date()} ({t * 365:.0f} jours)" for e, t in zip(svi_params["expiry"], svi_params["T"])]
    choice = st.selectbox("Maturité", range(len(labels)), format_func=lambda i: labels[i], index=6)
    row = svi_params.iloc[choice]
    T_sel = row["T"]
    p_sel = row[["a", "b", "rho", "m", "sigma"]].values.astype(float)
    s = smiles[smiles["expiry"] == row["expiry"]]
    k_fine = np.linspace(s["k"].min(), s["k"].max(), 120)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=s["k"], y=s["iv"] * 100, mode="markers", name="Marché", marker=dict(size=5)))
    fig_s.add_trace(go.Scatter(x=k_fine, y=svi.svi_implied_vol(k_fine, T_sel, p_sel) * 100, name="SVI"))
    if T_sel >= 30 / 365:
        F = s["F"].iloc[0]
        # Heston affiché uniquement sur sa zone de calibration
        atm_sel = svi.svi_implied_vol(0.0, T_sel, p_sel)
        k_h = np.linspace(-2.5 * atm_sel * np.sqrt(T_sel), 1.5 * atm_sel * np.sqrt(T_sel), 60)
        prices = heston.heston_call(F, F * np.exp(k_h), T_sel, r_mkt,
                                    hp["v0"], hp["kappa"], hp["theta"], hp["xi"], hp["rho"])
        iv_h = [implied_vol(pr, F, F * np.exp(kk), T_sel, r_mkt, "call", q=r_mkt) for pr, kk in zip(prices, k_h)]
        fig_s.add_trace(go.Scatter(x=k_h, y=np.array(iv_h) * 100, name="Heston (zone de calibration)",
                                   line=dict(dash="dash")))
    else:
        st.caption("Heston n'est pas affiché : il a été calibré sur les maturités de 30 jours et plus.")
    fig_s.update_layout(xaxis_title="Log-moneyness k = ln(K/F)", yaxis_title="Vol implicite (%)", height=450)
    st.plotly_chart(fig_s)


# =============================================================
# Onglet 3 : Variance swap
# =============================================================
@st.cache_data
def compute_vs_table(date):
    smiles, svi_params, hp, _, _ = load_snapshot_results(date)
    rows = []
    for _, row in svi_params.iterrows():
        T = row["T"]
        p = row[["a", "b", "rho", "m", "sigma"]].values.astype(float)
        s = smiles[smiles["expiry"] == row["expiry"]]
        rows.append({
            "Échéance": row["expiry"].date(),
            "Jours": round(T * 365),
            "T": T,
            "Vol ATM": 100 * svi.svi_implied_vol(0.0, T, p),
            "VS marché": 100 * np.sqrt(vs.var_swap_strike_svi(T, p)),
            "VS tronqué": 100 * np.sqrt(vs.var_swap_strike_svi(T, p, k_min=s["k"].min(), k_max=s["k"].max())),
            "VS Heston": 100 * np.sqrt(vs.var_swap_strike_heston(T, hp["v0"], hp["kappa"], hp["theta"])),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=24 * 3600)
def official_vix(date):
    """Clôture officielle du VIX à la date de la photo (None si indisponible)."""
    try:
        import yfinance as yf
        end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        h = yf.Ticker("^VIX").history(start=date, end=end)["Close"]
        return float(h.iloc[0]) if len(h) else None
    except Exception:
        return None


with tab_varswap:
    date = config.SNAPSHOT_DATE
    smiles, svi_params, hp, S0, r_mkt = load_snapshot_results(date)
    vt = compute_vs_table(date)

    st.subheader("Variance swaps : le prix de la volatilité")
    st.markdown("Un variance swap paie la différence entre la variance **réalisée** et un strike fixé à l'avance. "
                "Ce strike se **réplique** avec toutes les options hors de la monnaie, pondérées par 1/K², "
                "ce qui permet de le calculer directement à partir de la surface de vol.")

    # Mon VIX : variance swap à 30 jours, interpolé comme le VIX officiel
    T30 = 30 / 365
    tv = (vt["VS tronqué"] / 100) ** 2 * vt["T"]
    near = vt[vt["T"] <= T30].index[-1]
    nxt = vt[vt["T"] > T30].index[0]
    w = (T30 - vt.loc[near, "T"]) / (vt.loc[nxt, "T"] - vt.loc[near, "T"])
    my_vix = 100 * np.sqrt((tv[near] + w * (tv[nxt] - tv[near])) / T30)
    off = official_vix(date)
    one_year = vt.iloc[(vt["T"] - 1).abs().argmin()]

    c = st.columns(3)
    c[0].metric("Mon VIX (reconstruit sur SPY)", f"{my_vix:.2f}")
    c[1].metric("VIX officiel", f"{off:.2f}" if off else "indisponible",
                delta=f"{my_vix - off:+.2f} d'écart" if off else None, delta_color="off")
    c[2].metric(f"Prime de skew ({one_year['Jours']} jours)",
                f"{one_year['VS marché'] - one_year['Vol ATM']:.1f} pts de vol")

    # Structure par terme
    fig_vs = go.Figure()
    fig_vs.add_trace(go.Scatter(x=vt["Jours"], y=vt["Vol ATM"], mode="lines+markers", name="Vol ATM"))
    fig_vs.add_trace(go.Scatter(x=vt["Jours"], y=vt["VS marché"], mode="lines+markers", name="Variance swap – marché (SVI)"))
    fig_vs.add_trace(go.Scatter(x=vt["Jours"], y=vt["VS Heston"], mode="lines+markers", name="Variance swap – Heston",
                                line=dict(dash="dash")))
    fig_vs.add_vline(x=30, line_dash="dot", annotation_text="Calibration Heston ≥ 30 j")
    fig_vs.update_layout(title="Strike de variance swap vs vol ATM", xaxis_title="Maturité (jours)",
                         yaxis_title="Volatilité (%)", height=450)
    st.plotly_chart(fig_vs)

    with st.expander("Voir le tableau complet"):
        st.dataframe(vt.drop(columns="T").round(2), hide_index=True)

    # D'où vient le prix ? Contribution de chaque strike
    st.markdown("#### D'où vient le prix d'un variance swap ?")
    labels = [f"{e.date()} ({t * 365:.0f} jours)" for e, t in zip(svi_params["expiry"], svi_params["T"])]
    idx_1y = int((svi_params["T"] - 1).abs().argmin())
    choice = st.selectbox("Maturité", range(len(labels)), format_func=lambda i: labels[i],
                          index=idx_1y, key="vs_maturity")
    row = svi_params.iloc[choice]
    T_sel = row["T"]
    p_sel = row[["a", "b", "rho", "m", "sigma"]].values.astype(float)
    atm = svi.svi_implied_vol(0.0, T_sel, p_sel)
    k = np.linspace(-6 * atm * np.sqrt(T_sel), 3 * atm * np.sqrt(T_sel), 600)
    w_k = np.maximum(svi.svi_total_variance(k, *p_sel), 1e-12)
    contrib = vs.otm_forward_price(k, w_k) * np.exp(-k)

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=k[k < 0], y=contrib[k < 0], fill="tozeroy", name="Puts (strikes sous le forward)"))
    fig_c.add_trace(go.Scatter(x=k[k >= 0], y=contrib[k >= 0], fill="tozeroy", name="Calls (strikes au-dessus)"))
    fig_c.update_layout(xaxis_title="Log-moneyness k", yaxis_title="Contribution au strike de variance",
                        height=400)
    st.plotly_chart(fig_c)

    share_puts = vs.var_swap_strike_svi(T_sel, p_sel, k_max=0.0) / vs.var_swap_strike_svi(T_sel, p_sel)
    st.markdown(f"Pour cette maturité, les **puts** représentent **{share_puts:.0%}** du prix du variance swap. "
                "Avec un skew négatif, ce sont eux qui le rendent plus cher que la vol ATM.")

    # =============================================================
# Onglet 4 : Delta-hedging
# =============================================================
@st.cache_data
def run_hedging_sim(sigma_imp, sigma_real, n_steps, cost, n_paths=5000):
    """Vente d'un call ATM 1 mois (S = K = 100) couvert en delta, sur des trajectoires simulées."""
    S0_, K_, T_, r_ = 100.0, 100.0, 1 / 12, 0.04
    paths = hd.simulate_gbm(S0_, 0.08, sigma_real, T_, n_steps, n_paths, seed=7)
    pnl = hd.delta_hedge_pnl(paths, K_, T_, r_, sigma_imp, cost=cost)
    formula = hd.gamma_pnl_approx(paths, K_, T_, r_, sigma_imp)
    premium = bs.bs_price(S0_, K_, T_, r_, sigma_imp, "call")
    expected = (premium - bs.bs_price(S0_, K_, T_, r_, sigma_real, "call")) * np.exp(r_ * T_)
    return pnl, formula, premium, expected


with tab_hedge:
    st.subheader("Simulateur de delta-hedging")
    st.markdown("Un trader **vend** un call à la monnaie à 1 mois (spot 100, strike 100) à la vol implicite choisie, "
                "puis le **couvre en delta**. Son P&L ne dépend pas de la direction du marché, "
                "mais de la volatilité que le marché va réellement **réaliser**.")

    c1, c2 = st.columns(2)
    sigma_imp_h = c1.slider("Vol implicite : vol à laquelle l'option est vendue (%)", 5, 60, 20) / 100
    sigma_real_h = c1.slider("Vol réalisée par le marché (%)", 5, 60, 20) / 100
    freq_label = c2.select_slider("Fréquence de couverture",
                                  options=["Hebdomadaire", "Quotidienne", "4 fois par jour", "Toutes les heures"],
                                  value="Quotidienne")
    n_steps_h = {"Hebdomadaire": 4, "Quotidienne": 21, "4 fois par jour": 84, "Toutes les heures": 336}[freq_label]
    cost_bp = c2.slider("Coûts de transaction (points de base)", 0, 20, 0)

    pnl, formula, premium, expected = run_hedging_sim(sigma_imp_h, sigma_real_h, n_steps_h, cost_bp / 10_000)

    m = st.columns(4)
    m[0].metric("Prime encaissée", f"{premium:.3f}")
    m[1].metric("P&L moyen", f"{pnl.mean():+.3f}", delta=f"attendu sans coûts : {expected:+.3f}", delta_color="off")
    m[2].metric("Écart-type du P&L", f"{pnl.std():.3f}")
    m[3].metric("Trajectoires gagnantes", f"{(pnl > 0).mean():.0%}")

    fig_h = go.Figure(go.Histogram(x=pnl, nbinsx=80))
    fig_h.add_vline(x=0, line_color="black")
    fig_h.add_vline(x=pnl.mean(), line_dash="dash", annotation_text="Moyenne")
    fig_h.update_layout(title="Distribution du P&L du vendeur couvert (5 000 trajectoires simulées)",
                        xaxis_title="P&L à maturité", yaxis_title="Nombre de trajectoires",
                        height=420, showlegend=False)
    st.plotly_chart(fig_h)

    if st.checkbox("Vérifier la formule ½·Γ·S²·(σ²_imp − σ²_réalisée)"):
        fig_f = go.Figure(go.Scatter(x=formula, y=pnl, mode="markers",
                                     marker=dict(size=3, opacity=0.4), name="Trajectoires"))
        lims = [min(formula.min(), pnl.min()), max(formula.max(), pnl.max())]
        fig_f.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="red"),
                                   name="Égalité parfaite"))
        fig_f.update_layout(xaxis_title="P&L selon la formule du gamma", yaxis_title="P&L réel de la couverture",
                            height=450)
        st.plotly_chart(fig_f)
        if cost_bp > 0:
            st.caption("Avec des coûts de transaction, les points passent sous la diagonale : la formule ne les inclut pas.")


# =============================================================
# Onglet 5 : Backtest
# =============================================================
@st.cache_data
def load_hist(end):
    return data.load_history("2007-01-01", end)


@st.cache_data
def run_backtest(end, vol_shift, cost):
    return backtest.monthly_vol_selling(load_hist(end), cost=cost, straddle_vol_shift=vol_shift)


with tab_backtest:
    st.subheader("Vendre de la volatilité sur SPY depuis 2007")
    st.markdown("Chaque mois, on vend la vol à 1 mois au niveau du **VIX**, de deux façons : "
                "un **variance swap**, et un **straddle à la monnaie** couvert en delta chaque jour "
                "sur les vrais cours de SPY.")

    c1, c2 = st.columns(2)
    shift = c1.slider("Vol de vente du straddle : VIX moins … points (2.7 ≈ vol ATM réaliste)",
                      0.0, 4.0, 2.7, 0.1)
    cost_bp_b = c2.slider("Coûts de couverture du straddle (points de base)", 0.0, 5.0, 0.0, 0.5)
    bt = run_backtest(config.SNAPSHOT_DATE, shift / 100, cost_bp_b / 10_000)

    sharpe = bt["straddle_pnl_pct"].mean() / bt["straddle_pnl_pct"].std() * np.sqrt(12)
    m = st.columns(4)
    m[0].metric("VIX moyen", f"{bt['vix'].mean():.1%}")
    m[1].metric("Vol réalisée moyenne", f"{bt['realized'].mean():.1%}")
    m[2].metric("Mois où implicite > réalisée", f"{(bt['vix'] > bt['realized']).mean():.0%}")
    m[3].metric("Sharpe du straddle (annualisé)", f"{sharpe:.2f}")

    fig_b = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                          subplot_titles=("Vol implicite (VIX) vs vol réalisée le mois suivant",
                                          "P&L cumulé : variance swap vendu (points de vol)",
                                          "P&L cumulé : straddle couvert vendu (% du spot)"))
    fig_b.add_trace(go.Scatter(x=bt.index, y=bt["vix"] * 100, name="VIX"), row=1, col=1)
    fig_b.add_trace(go.Scatter(x=bt.index, y=bt["realized"] * 100, name="Vol réalisée"), row=1, col=1)
    fig_b.add_trace(go.Scatter(x=bt.index, y=bt["varswap_pnl"].cumsum(), name="Variance swap",
                               showlegend=False), row=2, col=1)
    fig_b.add_trace(go.Scatter(x=bt.index, y=bt["straddle_pnl_pct"].cumsum(), name="Straddle",
                               showlegend=False), row=3, col=1)
    fig_b.update_layout(height=850)
    st.plotly_chart(fig_b)

    st.markdown("#### Les pires mois pour un vendeur de volatilité")
    worst = bt.nsmallest(5, "varswap_pnl")[["vix", "realized", "varswap_pnl", "straddle_pnl_pct"]].copy()
    worst.index = worst.index.date
    worst[["vix", "realized"]] *= 100
    worst.columns = ["VIX à la vente (%)", "Vol réalisée (%)", "P&L variance swap (pts)", "P&L straddle (% spot)"]
    st.dataframe(worst.round(2))
    st.caption("En février 2020, le straddle perd beaucoup moins que le variance swap : le marché s'est "
               "vite éloigné du strike, là où le gamma est faible. Le P&L du straddle est pondéré par le gamma, "
               "celui du variance swap ne l'est pas.")