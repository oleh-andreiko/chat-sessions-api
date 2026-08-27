"""Model prices and the cost formula.

Kept in its own module so the formula never ends up inside a route handler:
adding a model is one dictionary entry and touches nothing else. The module
knows nothing about the database or FastAPI, so it can be tested on its own.

Rates are USD per 1M tokens, taken by hand from the official OpenAI price list
(the API does not serve prices). Checked on 2026-08-27.

Prices are keyed by the alias a caller asks for, not by the dated snapshot
OpenAI answers with: the published price list quotes aliases, and no dated name
would ever be found here.
"""

from decimal import Decimal

from app.errors import PricingNotConfigured

TOKENS_PER_UNIT = Decimal("1000000")

MODEL_PRICING = {
    "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "gpt-4.1-nano": {"input": Decimal("0.10"), "output": Decimal("0.40")},
    "gpt-4.1-mini": {"input": Decimal("0.40"), "output": Decimal("1.60")},
}

SUPPORTED_MODELS = sorted(MODEL_PRICING)


def is_supported(model: str) -> bool:
    return model in MODEL_PRICING


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    prices = MODEL_PRICING.get(model)
    if prices is None:
        # Deliberately loud. Falling back to zero would silently report a paid
        # session as free, which is worse than a failed request.
        raise PricingNotConfigured(f"No pricing configured for model '{model}'.")

    return (
        Decimal(prompt_tokens) / TOKENS_PER_UNIT * prices["input"]
        + Decimal(completion_tokens) / TOKENS_PER_UNIT * prices["output"]
    )
