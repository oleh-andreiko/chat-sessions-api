"""Tests for the cost formula.

pricing.py is a pure function over a dictionary: no database, no network, no
FastAPI. That is the whole point of keeping it in its own module — these tests
need no fixtures and no mocking.
"""

from decimal import Decimal

import pytest

from app.errors import PricingNotConfigured
from app.services.pricing import MODEL_PRICING, calculate_cost

MODEL = "gpt-4o-mini"


def test_rates_match_the_published_price_list():
    # $0.15 / $1M input and $0.60 / $1M output, checked against the official
    # OpenAI price list on 2026-08-25 and quoted in the README.
    assert MODEL_PRICING[MODEL] == {
        "input": Decimal("0.15"),
        "output": Decimal("0.60"),
    }


def test_one_million_tokens_each_way():
    # The easiest case to verify by hand: 0.15 + 0.60.
    assert calculate_cost(MODEL, 1_000_000, 1_000_000) == Decimal("0.75")


def test_input_and_output_are_priced_differently():
    # Output costs four times input, so swapping the two must change the result.
    assert calculate_cost(MODEL, 1_000_000, 0) == Decimal("0.15")
    assert calculate_cost(MODEL, 0, 1_000_000) == Decimal("0.60")


def test_a_real_exchange():
    # Numbers taken from an actual reply logged in the README example.
    assert calculate_cost(MODEL, 42, 12) == Decimal("0.00001350")


def test_empty_usage_costs_nothing():
    assert calculate_cost(MODEL, 0, 0) == Decimal("0")


def test_result_is_decimal_not_float():
    # float would accumulate binary rounding error on money.
    assert isinstance(calculate_cost(MODEL, 7, 3), Decimal)


def test_small_amounts_keep_their_precision():
    # A single input token costs $0.00000015. Rounding this to the cent, or to
    # six decimal places, would erase it.
    assert calculate_cost(MODEL, 1, 0) == Decimal("0.00000015")


def test_unknown_model_raises_instead_of_returning_zero():
    # A silent zero would report a paid session as free.
    with pytest.raises(PricingNotConfigured):
        calculate_cost("gpt-does-not-exist", 100, 100)
