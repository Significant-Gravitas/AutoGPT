from __future__ import annotations

import pytest

from .model_pricing import ModelRate, compute_cost_usd, execution_path_discount


def test_unknown_model_returns_none():
    """Unknown model → None so callers can flag the cost as unknown
    rather than silently bill at zero."""
    assert (
        compute_cost_usd(
            model="totally-fake-model-xyz", input_tokens=100, output_tokens=50
        )
        is None
    )


def test_known_model_basic_math():
    """Sonnet 4.6: $3/M input + $15/M output. Spot check arithmetic."""
    cost = compute_cost_usd(
        model="claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(3.0 + 15.0)


def test_cache_tokens_priced_separately():
    """Cache reads + cache writes use their own per-Mtok rates."""
    # Sonnet 4.6 cache_read=$0.30/M, cache_write=$3.75/M
    cost = compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.30 + 3.75)


@pytest.mark.parametrize(
    "path,expected_discount",
    [
        ("sync_baseline", 0.0),
        ("anthropic_batch", 0.5),
        ("openai_batch", 0.5),
    ],
)
def test_execution_path_discount(path, expected_discount):
    assert execution_path_discount(path) == expected_discount


def test_batch_discount_applied_to_final_cost():
    """Batch path multiplies the rate-card cost by (1 - 0.5) = 0.5."""
    sync = compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        execution_path="sync_baseline",
    )
    batch = compute_cost_usd(
        model="claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        execution_path="anthropic_batch",
    )
    assert sync == pytest.approx(18.0)
    assert batch == pytest.approx(9.0)
    assert batch == pytest.approx(sync * 0.5)


def test_model_with_no_cache_pricing_ignores_cache_tokens():
    """gpt-5 has no cache_*_per_mtok — cache tokens contribute zero."""
    cost = compute_cost_usd(
        model="gpt-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
    )
    assert cost == pytest.approx(0.0)


def test_modelrate_is_frozen():
    """ModelRate is frozen so a typo in calling code can't mutate the
    shared rate card by accident."""
    rate = ModelRate(input_per_mtok=1.0, output_per_mtok=2.0)
    with pytest.raises(Exception):
        rate.input_per_mtok = 99.0  # type: ignore[misc]


def test_model_lookup_is_case_insensitive():
    """We don't want a registry miss because a caller passes the model
    name with the wrong case."""
    a = compute_cost_usd(model="claude-sonnet-4-6", input_tokens=100, output_tokens=100)
    b = compute_cost_usd(model="CLAUDE-SONNET-4-6", input_tokens=100, output_tokens=100)
    assert a is not None and b is not None
    assert a == pytest.approx(b)
