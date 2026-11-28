"""A fake Replicate HTTP API, so tests drive the real SDK through httpx.MockTransport."""

import json
from typing import Any

import httpx
import pytest
from replicate.client import Client as RealReplicateClient

NOT_FOUND = {"detail": "The requested resource could not be found.", "status": 404}


class FakeReplicate:
    """Answers the Replicate endpoints the block calls and records requests."""

    def __init__(self) -> None:
        self.models: dict[str, dict] = {}
        self.versions: dict[str, set[str]] = {}
        self.requests: list[tuple[str, str, dict]] = []
        self.status = "succeeded"
        self.error: str | None = None
        self.output: Any = "a cat on a sofa"
        self.metrics: dict | None = {"predict_time": 1.5}
        self.fail_with: tuple[int, dict] | None = None

    def add_model(
        self,
        name: str,
        *,
        official: bool,
        latest_version: str | None = None,
        versions: tuple[str, ...] = (),
    ) -> None:
        owner, model = name.split("/")
        self.models[name] = {
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
            "is_official": official,
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
        self.versions[name] = {*versions, *([latest_version] if latest_version else [])}

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((request.method, path, body))

        if self.fail_with:
            return httpx.Response(self.fail_with[0], json=self.fail_with[1])
        if request.method == "POST" and path == "/v1/predictions":
            return httpx.Response(201, json=self._prediction("", body))
        if request.method == "POST" and path.endswith("/predictions"):
            name = path.removeprefix("/v1/models/").removesuffix("/predictions")
            model = self.models.get(name)
            if model is None or not model["is_official"]:
                return httpx.Response(404, json=NOT_FOUND)
            return httpx.Response(201, json=self._prediction(name, body))

        if request.method == "GET" and path.startswith("/v1/models/"):
            name, _, version = path.removeprefix("/v1/models/").partition("/versions/")
            if version:
                if version not in self.versions.get(name, set()):
                    return httpx.Response(404, json=NOT_FOUND)
                return httpx.Response(200, json={"id": version, "openapi_schema": {}})
            model = self.models.get(name)
            if model is None:
                return httpx.Response(404, json=NOT_FOUND)
            return httpx.Response(200, json=model)

        return httpx.Response(500, json={"detail": f"unexpected {path}"})

    def predictions_created(self) -> list[str]:
        return [p for m, p, _ in self.requests if m == "POST"]

    def _prediction(self, model: str, body: dict) -> dict:
        return {
            "id": "pred-1",
            "model": model,
            "version": body.get("version", ""),
            "status": self.status,
            "input": body.get("input", {}),
            "output": self.output,
            "logs": "",
            "error": self.error,
            "metrics": self.metrics,
            "urls": {},
        }


@pytest.fixture
def fake_replicate(monkeypatch) -> FakeReplicate:
    fake = FakeReplicate()

    def build(**kwargs):
        return RealReplicateClient(
            transport=httpx.MockTransport(fake.handler), **kwargs
        )

    monkeypatch.setattr(
        "backend.blocks.replicate.replicate_block.ReplicateClient", build
    )
    return fake
