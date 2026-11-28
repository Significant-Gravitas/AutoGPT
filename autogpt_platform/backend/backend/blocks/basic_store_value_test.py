"""StoreValueBlock must return its stored data whenever it is not None."""

import pytest

from backend.blocks.basic import StoreValueBlock


async def _collect_outputs(block, input_data):
    return [item async for item in block.run(input_data)]


@pytest.mark.parametrize(
    ("stored", "trigger"),
    [
        pytest.param(False, "trigger value", id="data-false"),
        pytest.param(0, "trigger value", id="data-zero"),
        pytest.param("", "trigger value", id="data-empty-string"),
        pytest.param([], "trigger value", id="data-empty-list"),
        pytest.param({}, "trigger value", id="data-empty-dict"),
    ],
)
async def test_store_value_block_returns_falsy_stored_data(stored, trigger):
    """The trigger value is only used when ``data`` is None, per the field docs.

    A falsy stored value (0, False, "", [], {}) is still a stored value: the
    block must emit it instead of falling back to the trigger input.
    """
    block = StoreValueBlock()

    outputs = await _collect_outputs(block, block.Input(input=trigger, data=stored))

    assert outputs == [("output", stored)]


async def test_store_value_block_uses_trigger_only_without_data():
    """With no stored data, the block emits the trigger input, as documented."""
    block = StoreValueBlock()

    outputs = await _collect_outputs(
        block, block.Input(input="trigger value", data=None)
    )

    assert outputs == [("output", "trigger value")]
