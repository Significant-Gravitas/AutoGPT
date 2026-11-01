"""Tests for running ReplicateModelBlock without a version hash.

Replicate's model-level endpoint (``POST /v1/models/{owner}/{name}/predictions``)
only serves official models; community models answer 404 there. The HTTP
layer is mocked with an ``httpx.MockTransport`` so the real Replicate SDK
builds the requests:

1. Community model, no version → resolves ``latest_version`` and runs it
2. Official model, no version → runs through the model endpoint, no lookup
3. Unknown model → clear input error naming the model
4. Model with no published version → clear input error naming the model
5. A non-404 error from the model endpoint is not masked by the fallback
"""

import json

import httpx
import pytest
from replicate.client import Client as RealReplicateClient

from backend.blocks.replicate._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.replicate.replicate_block import ReplicateModelBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError

COMMUNITY_MODEL = "cuuupid/glm-4v-9b"
OFFICIAL_MODEL = "black-forest-labs/flux-schnell"
LATEST_VERSION = "69196a53d2a33ec4b1e4e3e8ee32d2e1a3a3f8e6e5d4c3b2a1f0e9d8c7b6a5f4"

NOT_FOUND = {"detail": "The requested resource could not be found.", "status": 404}


def _model_json(name: str, latest_version: str | None) -> dict:
    owner, model = name.split("/")
    return {
        "url": f"https://replicate.com/{name}",
        "owner": owner,
        "name": model,
        "description": None,
        "visibility": "public",
        "github_url": None,
        "paper_url": None,
        "license_url": None,
        "run_count": 1,
        "cover_image_url": None,
        "default_example": None,
        "latest_version": (
            {
                "id": latest_version,
                "created_at": "2024-06-12T10:00:00.000000Z",
                "cog_version": "0.9.9",
                "openapi_schema": {},
            }
            if latest_version
            else None
        ),
    }


def _prediction_json(model: str, version: str) -> dict:
    return {
        "id": "pred-1",
        "model": model,
        "version": version,
        "status": "succeeded",
        "input": {},
        "output": "a cat on a sofa",
        "logs": "",
        "error": None,
        "metrics": {"predict_time": 1.5},
        "urls": {},
    }


class FakeReplicate:
    """Answers the Replicate endpoints the block calls and records requests."""

    def __init__(self, models: dict[str, dict | None], official: set[str]):
        self.models = models
        self.official = official
        self.requests: list[tuple[str, str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((request.method, path, body))

        if request.method == "POST" and path.endswith("/predictions"):
            if path == "/v1/predictions":
                return httpx.Response(201, json=_prediction_json("", body["version"]))
            name = path.removeprefix("/v1/models/").removesuffix("/predictions")
            if name in self.official:
                return httpx.Response(201, json=_prediction_json(name, ""))
            return httpx.Response(404, json=NOT_FOUND)

        if request.method == "GET" and path.startswith("/v1/models/"):
            name = path.removeprefix("/v1/models/")
            model = self.models.get(name)
            if model is None:
                return httpx.Response(404, json=NOT_FOUND)
            return httpx.Response(200, json=model)

        return httpx.Response(500, json={"detail": f"unexpected {path}"})


def _patch_client(monkeypatch, fake: FakeReplicate):
    def build(**kwargs):
        return RealReplicateClient(
            transport=httpx.MockTransport(fake.handler), **kwargs
        )

    monkeypatch.setattr(
        "backend.blocks.replicate.replicate_block.ReplicateClient", build
    )


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
async def test_community_model_without_version_runs_latest_version(monkeypatch):
    fake = FakeReplicate(
        models={COMMUNITY_MODEL: _model_json(COMMUNITY_MODEL, LATEST_VERSION)},
        official=set(),
    )
    _patch_client(monkeypatch, fake)

    outputs = await _run(COMMUNITY_MODEL)

    assert outputs == {
        "result": "a cat on a sofa",
        "status": "succeeded",
        "model_name": COMMUNITY_MODEL,
    }
    assert ("GET", f"/v1/models/{COMMUNITY_MODEL}", {}) in fake.requests
    method, path, body = fake.requests[-1]
    assert (method, path) == ("POST", "/v1/predictions")
    assert body["version"] == LATEST_VERSION
    assert body["input"] == {"prompt": "describe"}


@pytest.mark.asyncio
async def test_official_model_without_version_uses_model_endpoint(monkeypatch):
    fake = FakeReplicate(
        models={OFFICIAL_MODEL: _model_json(OFFICIAL_MODEL, None)},
        official={OFFICIAL_MODEL},
    )
    _patch_client(monkeypatch, fake)

    outputs = await _run(OFFICIAL_MODEL)

    assert outputs["result"] == "a cat on a sofa"
    assert [(m, p) for m, p, _ in fake.requests] == [
        ("POST", f"/v1/models/{OFFICIAL_MODEL}/predictions")
    ]


@pytest.mark.asyncio
async def test_pinned_version_skips_model_endpoint(monkeypatch):
    fake = FakeReplicate(models={}, official=set())
    _patch_client(monkeypatch, fake)

    await _run(COMMUNITY_MODEL, version=LATEST_VERSION)

    assert [(m, p) for m, p, _ in fake.requests] == [("POST", "/v1/predictions")]
    assert fake.requests[0][2]["version"] == LATEST_VERSION


@pytest.mark.asyncio
async def test_unknown_model_gives_clear_input_error(monkeypatch):
    fake = FakeReplicate(models={}, official=set())
    _patch_client(monkeypatch, fake)

    with pytest.raises(BlockInputError) as exc_info:
        await _run("nobody/does-not-exist")

    message = str(exc_info.value)
    assert "'nobody/does-not-exist'" in message
    assert "not found" in message


@pytest.mark.asyncio
async def test_model_without_versions_gives_clear_input_error(monkeypatch):
    fake = FakeReplicate(
        models={COMMUNITY_MODEL: _model_json(COMMUNITY_MODEL, None)},
        official=set(),
    )
    _patch_client(monkeypatch, fake)

    with pytest.raises(BlockInputError) as exc_info:
        await _run(COMMUNITY_MODEL)

    message = str(exc_info.value)
    assert f"'{COMMUNITY_MODEL}'" in message
    assert "no published version" in message
    assert not any(p == "/v1/predictions" for _, p, _ in fake.requests)


@pytest.mark.asyncio
async def test_non_404_error_is_not_retried_as_community_model(monkeypatch):
    class Unauthorized(FakeReplicate):
        def handler(self, request: httpx.Request) -> httpx.Response:
            self.requests.append((request.method, request.url.path, {}))
            return httpx.Response(401, json={"detail": "Invalid token", "status": 401})

    fake = Unauthorized(models={}, official=set())
    _patch_client(monkeypatch, fake)

    with pytest.raises(BlockExecutionError) as exc_info:
        await _run(COMMUNITY_MODEL)

    assert "Invalid token" in str(exc_info.value)
    assert [(m, p) for m, p, _ in fake.requests] == [
        ("POST", f"/v1/models/{COMMUNITY_MODEL}/predictions")
    ]
