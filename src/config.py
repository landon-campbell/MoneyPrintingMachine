from pathlib import Path
import os
from dotenv import load_dotenv

# Paths
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Instruments
TICKERS = {
    "SP500":  "^GSPC",   # S&P 500
    "NASDAQ": "^IXIC",   # NASDAQ Composite
    "DOW":    "^DJI",    # Dow Jones IA
    # Extend freely, e.g.  "AAPL": "AAPL"
}

# Alpaca integration
USE_ALPACA = bool(os.getenv("PRISM_USE_ALPACA", True))

if USE_ALPACA:
    load_dotenv()                    # .env file
    ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError(
            "PRISM_USE_ALPACA is True but ALPACA_API_KEY / SECRET not set."
        )
