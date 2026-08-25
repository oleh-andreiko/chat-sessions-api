"""Settings, read once from .env at import time.

Plain module-level constants instead of a settings class: the app has a handful
of knobs and nothing needs to swap them at runtime.
"""

import os

from dotenv import load_dotenv

load_dotenv()



def _required(name: str) -> str:
    """Fail at startup with an instruction rather than a KeyError later.

    An empty value counts as missing: .env.example ships with an empty
    OPENAI_API_KEY, so forgetting to fill it in is the likeliest mistake.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


DATABASE_URL = _required("DATABASE_URL")
OPENAI_API_KEY = _required("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# How many past messages are replayed to the model. Caps both the cost of each
# request and the risk of overflowing the model's context window.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
