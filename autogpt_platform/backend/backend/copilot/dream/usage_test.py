"""A batch pass's usage, read off the phase rows its callbacks kept."""

import pytest

from .usage import batch_pass_usage

_MODELS = {
    "consolidate": "claude-sonnet-5",
    "recombine": "claude-opus-5-5",
    "sanitize": "claude-sonnet-5",
}
# Catalog list prices per million tokens, halved on the batch route.
_SONNET_5 = (10 * 2.0 + 20 * 10.0) / 1_000_000 / 2
_OPUS_5_5 = (10 * 4.0 + 20 * 20.0) / 1_000_000 / 2


def _landed(**overrides) -> dict:
    return {"input_tokens": 10, "output_tokens": 20, "error": None, **overrides}


def test_each_landed_phase_is_priced_on_its_own_model_at_the_batch_discount():
    usage = batch_pass_usage(
        {"consolidate": _landed(), "recombine": _landed(), "sanitize": _landed()},
        _MODELS,
    )

    assert usage is not None
    assert [(p.phase, p.model) for p in usage.phases] == list(_MODELS.items())
    assert [p.cost_usd for p in usage.phases] == pytest.approx(
        [_SONNET_5, _OPUS_5_5, _SONNET_5]
    )
    assert usage.total_cost_usd == pytest.approx(2 * _SONNET_5 + _OPUS_5_5)
    assert (usage.total_input_tokens, usage.total_output_tokens) == (30, 60)
    assert usage.discount_applied == 0.5


def test_errored_phases_and_phases_without_a_model_are_left_out():
    usage = batch_pass_usage(
        {
            "consolidate": _landed(),
            "recombine": _landed(error="provider down"),
            "sanitize": _landed(),
        },
        {"consolidate": "claude-sonnet-5", "recombine": "claude-opus-5-5"},
    )

    assert usage is not None
    assert [p.phase for p in usage.phases] == ["consolidate"]


def test_a_phase_that_reported_no_tokens_keeps_an_unknown_cost():
    """As ``inference.record.record`` leaves it: unknown, not a known $0."""
    usage = batch_pass_usage(
        {
            "consolidate": _landed(),
            "recombine": _landed(input_tokens=0, output_tokens=0),
        },
        _MODELS,
    )

    assert usage is not None
    assert [p.cost_usd for p in usage.phases] == [pytest.approx(_SONNET_5), None]
    assert usage.total_cost_usd is None


def test_nothing_landed_is_no_usage():
    assert batch_pass_usage({}, _MODELS) is None
    assert batch_pass_usage({"consolidate": _landed(error="boom")}, _MODELS) is None
