"""Tests for which Replicate model and version ReplicateModelBlock runs.

Replicate's model-level endpoint (``POST /v1/models/{owner}/{name}/predictions``)
only serves official models; community models answer 404 there. The HTTP
layer is faked (``conftest.FakeReplicate``) so the real Replicate SDK builds
the requests:

1. Community model, no version → resolves ``latest_version`` and runs it
2. Official model, no version → runs through the model endpoint, no lookup
3. A pinned version runs only if it belongs to the named model
4. Unknown model → clear input error naming the model
5. Model with no published version → clear input error naming the model
6. A non-404 error from the model endpoint is not masked by the fallback
"""

import pytest

from backend.blocks.replicate._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.replicate.replicate_block import ReplicateModelBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError

COMMUNITY_MODEL = "cuuupid/glm-4v-9b"
OFFICIAL_MODEL = "black-forest-labs/flux-schnell"
LATEST_VERSION = "69196a53d2a33ec4b1e4e3e8ee32d2e1a3a3f8e6e5d4c3b2a1f0e9d8c7b6a5f4"
OTHER_VERSION = "74cdc46ff77a5401231e2491ab55c431a53157c7b726bde6e66fa565d60587ab"


async def _run(model_name: str, version: str | None = None) -> dict:
    block = ReplicateModelBlock()
    input_data = ReplicateModelBlock.Input(
        **{
            "credentials": TEST_CREDENTIALS_INPUT,
            "model_name": model_name,
            "model_inputs": {"prompt": "describe"},
            "version": version,
        }
    )
    outputs = {}
    async for name, value in block.run(
        input_data,
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(user_id="user-1", graph_exec_id="exec-1"),
    ):
        outputs[name] = value
    return outputs


@pytest.mark.asyncio
async def test_community_model_without_version_runs_latest_version(fake_replicate):
    fake_replicate.add_model(
        COMMUNITY_MODEL, official=False, latest_version=LATEST_VERSION
    )

    outputs = await _run(COMMUNITY_MODEL)

    assert outputs == {
        "result": "a cat on a sofa",
        "status": "succeeded",
        "model_name": COMMUNITY_MODEL,
    }
    assert [(m, p) for m, p, _ in fake_replicate.requests] == [
        ("GET", f"/v1/models/{COMMUNITY_MODEL}"),
        ("POST", "/v1/predictions"),
    ]
    body = fake_replicate.requests[-1][2]
    assert body["version"] == LATEST_VERSION
    assert body["input"] == {"prompt": "describe"}


@pytest.mark.asyncio
async def test_official_model_without_version_uses_model_endpoint(fake_replicate):
    fake_replicate.add_model(OFFICIAL_MODEL, official=True)

    outputs = await _run(OFFICIAL_MODEL)

    assert outputs["result"] == "a cat on a sofa"
    assert [(m, p) for m, p, _ in fake_replicate.requests] == [
        ("POST", f"/v1/models/{OFFICIAL_MODEL}/predictions")
    ]


@pytest.mark.asyncio
async def test_pinned_version_of_the_named_model_runs_it(fake_replicate):
    fake_replicate.add_model(
        COMMUNITY_MODEL, official=False, latest_version=LATEST_VERSION
    )

    await _run(COMMUNITY_MODEL, version=LATEST_VERSION)

    assert fake_replicate.predictions_created() == ["/v1/predictions"]
    assert fake_replicate.requests[-1][2]["version"] == LATEST_VERSION


@pytest.mark.asyncio
async def test_pinned_version_of_another_model_is_refused(fake_replicate):
    fake_replicate.add_model(
        COMMUNITY_MODEL, official=False, latest_version=LATEST_VERSION
    )

    with pytest.raises(BlockInputError) as exc_info:
        await _run(COMMUNITY_MODEL, version=OTHER_VERSION)

    assert f"'{OTHER_VERSION}' was not found" in str(exc_info.value)
    assert fake_replicate.predictions_created() == []


@pytest.mark.asyncio
async def test_unknown_model_gives_clear_input_error(fake_replicate):
    with pytest.raises(BlockInputError) as exc_info:
        await _run("nobody/does-not-exist")

    message = str(exc_info.value)
    assert "'nobody/does-not-exist'" in message
    assert "not found" in message


@pytest.mark.asyncio
async def test_model_without_versions_gives_clear_input_error(fake_replicate):
    fake_replicate.add_model(COMMUNITY_MODEL, official=False)

    with pytest.raises(BlockInputError) as exc_info:
        await _run(COMMUNITY_MODEL)

    message = str(exc_info.value)
    assert f"'{COMMUNITY_MODEL}'" in message
    assert "no published version" in message
    assert fake_replicate.predictions_created() == []


@pytest.mark.asyncio
async def test_non_404_error_is_not_retried_as_community_model(fake_replicate):
    fake_replicate.fail_with = (401, {"detail": "Invalid token", "status": 401})

    with pytest.raises(BlockExecutionError) as exc_info:
        await _run(OFFICIAL_MODEL)

    assert "Invalid token" in str(exc_info.value)
    assert [(m, p) for m, p, _ in fake_replicate.requests] == [
        ("POST", f"/v1/models/{OFFICIAL_MODEL}/predictions")
    ]
