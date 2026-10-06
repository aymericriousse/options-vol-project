import pandas as pd
import yfinance as yf
from .config import DATA_DIR


def get_spot(ticker="SPY"):
    """Dernier prix de clôture (ou prix courant) du sous-jacent."""
    hist = yf.Ticker(ticker).history(period="5d")
    return float(hist["Close"].iloc[-1])


def get_risk_free_rate():
    """Taux sans risque approché par le T-bill US à 13 semaines (^IRX), en décimal."""
    hist = yf.Ticker("^IRX").history(period="5d")
    return float(hist["Close"].iloc[-1]) / 100


def download_option_chain(ticker="SPY"):
    """Télécharge toutes les options (calls et puts) de toutes les maturités."""
    tkr = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    frames = []

    for exp in tkr.options:
        chain = tkr.option_chain(exp)
        for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df.copy()
            df["type"] = opt_type
            df["expiry"] = pd.Timestamp(exp)
            frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["T"] = (data["expiry"] - today).dt.days / 365.0   # maturité en années
    data["mid"] = (data["bid"] + data["ask"]) / 2           # prix milieu bid/ask

    cols = ["type", "expiry", "T", "strike", "bid", "ask", "mid",
            "lastPrice", "volume", "openInterest", "impliedVolatility"]
    return data[cols]


def clean_chain(df, min_T=7 / 365, max_rel_spread=0.5):
    """Filtre les options inexploitables."""
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]   # prix cotés et cohérents
    df = df[df["T"] >= min_T]                               # on retire les maturités < 7 jours
    rel_spread = (df["ask"] - df["bid"]) / df["mid"]
    df = df[rel_spread <= max_rel_spread]                   # on retire les options trop illiquides
    return df.reset_index(drop=True)

def snapshot_paths(date, ticker="SPY"):
    """Chemins des fichiers d'une photo du marché."""
    t = ticker.lower()
    return (DATA_DIR / f"{t}_options_{date}.csv",
            DATA_DIR / f"{t}_market_{date}.csv",
            DATA_DIR / f"{t}_options_raw_{date}.csv")


def create_snapshot(ticker="SPY"):
    """Télécharge et sauvegarde une photo du marché du jour. Refuse d'écraser."""
    date = pd.Timestamp.today().strftime("%Y-%m-%d")
    opt_path, mkt_path, raw_path = snapshot_paths(date, ticker)
    if opt_path.exists():
        raise FileExistsError(f"Une photo du {date} existe déjà ({opt_path.name}).")

    DATA_DIR.mkdir(exist_ok=True)
    S0, r = get_spot(ticker), get_risk_free_rate()
    raw = download_option_chain(ticker)
    clean = clean_chain(raw)

    raw.to_csv(raw_path, index=False)
    clean.to_csv(opt_path, index=False)
    pd.Series({"date": date, "spot": S0, "r": r}).to_csv(mkt_path)

    print(f"Photo du {date} créée : spot = {S0:.2f}, r = {r:.2%}, "
          f"{len(raw)} options brutes, {len(clean)} après nettoyage")
    return date


def load_snapshot(date, ticker="SPY"):
    """Charge une photo du marché déjà sauvegardée."""
    opt_path, mkt_path, _ = snapshot_paths(date, ticker)
    chain = pd.read_csv(opt_path, parse_dates=["expiry"])
    market = pd.read_csv(mkt_path, index_col=0).iloc[:, 0]
    return chain, float(market["spot"]), float(market["r"])