from pathlib import Path
import os
from dotenv import load_dotenv

ROOT      = Path(__file__).parent.parent.resolve()
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "SP500":  "SPY",
    "NASDAQ": "QQQ",
    "DOW":    "DIA",
}

# look for .env in project root
load_dotenv(dotenv_path=ROOT / ".env")

USE_ALPACA   = os.getenv("PRISM_USE_ALPACA", "false").lower() == "true"
ALPACA_KEY   = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET= os.getenv("ALPACA_API_SECRET")

if USE_ALPACA and not (ALPACA_KEY and ALPACA_SECRET):
    raise RuntimeError("PRISM_USE_ALPACA is true but API keys not found.")
