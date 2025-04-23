from pathlib import Path
import os
from dotenv import load_dotenv

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
DATA = ROOT / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"
for p in (RAW, PROC):
    p.mkdir(parents=True, exist_ok=True)

# ── load .env next to this file ──────────────────────────────────────────
load_dotenv(ROOT / ".env")

# ── Alpaca configuration ────────────────────────────────────────────────
# Users may supply either the classic Alpaca names
#   APCA_API_KEY_ID / APCA_API_SECRET_KEY
# or the earlier PRISM examples
#   ALPACA_API_KEY   / ALPACA_API_SECRET
#
ALPACA_KEY = (
    os.getenv("APCA_API_KEY_ID")          # preferred names
    or os.getenv("ALPACA_API_KEY")        # fallback (legacy)
)
ALPACA_SEC = (
    os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_API_SECRET")
)

USE_ALPACA = os.getenv("PRISM_USE_ALPACA", "true").lower() == "true"

if USE_ALPACA and not (ALPACA_KEY and ALPACA_SEC):
    raise RuntimeError(
        "PRISM_USE_ALPACA=true but no Alpaca credentials found.\n"
        "Set APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env "
        "or disable intraday analysis with PRISM_USE_ALPACA=false."
    )
