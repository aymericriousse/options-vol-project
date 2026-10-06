from pathlib import Path

# Date de la photo du marché utilisée dans TOUT le projet.
SNAPSHOT_DATE = "2026-10-06"
TICKER = "SPY"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"