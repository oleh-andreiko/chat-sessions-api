"""Settings, read once from .env at import time.

Plain module-level constants instead of a settings class: the app has a handful
of knobs and nothing needs to swap them at runtime.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# How many past messages are replayed to the model. Caps both the cost of each
# request and the risk of overflowing the model's context window.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
