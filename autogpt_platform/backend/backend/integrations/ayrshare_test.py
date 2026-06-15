from __future__ import annotations

import importlib
import sys
import types

import pytest


class _Requests:
    pass


request_module = types.ModuleType("backend.util.request")
request_module.Requests = _Requests
sys.modules.setdefault("backend.util.request", request_module)

ayrshare = importlib.import_module("backend.integrations.ayrshare")
AyrshareClient = ayrshare.AyrshareClient
SocialPlatform = ayrshare.SocialPlatform


class _Response:
    ok = True
    status = 200

    def __init__(self, data: dict):
        self._data = data

    def json(self) -> dict:
        return self._data


class _FakeRequests:
    def __init__(self, response: _Response):
        self.response = response
        self.calls: list[dict] = []

    async def post(self, url: str, *, json: dict, headers: dict | None = None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        return self.response


def _set_ayrshare_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.integrations.ayrshare.settings.secrets.ayrshare_api_key",
        "org-api-key",
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_jwt_strips_profile_key_without_mutating_base_headers(monkeypatch):
    _set_ayrshare_api_key(monkeypatch)
    requests = _FakeRequests(
        _Response(
            {
                "status": "success",
                "title": "Profile",
                "token": "jwt",
                "url": "https://profile.test/sso",
            }
        )
    )
    client = AyrshareClient(custom_requests=requests)

    await client.generate_jwt(private_key="private-key", profile_key=" profile-key ")

    call = requests.calls[0]
    assert call["json"]["profileKey"] == "profile-key"
    assert call["headers"]["Profile-Key"] == "profile-key"
    assert "Profile-Key" not in client.headers


@pytest.mark.asyncio(loop_scope="session")
async def test_create_post_strips_profile_key_without_mutating_base_headers(monkeypatch):
    _set_ayrshare_api_key(monkeypatch)
    requests = _FakeRequests(
        _Response(
            {
                "status": "success",
                "posts": [
                    {
                        "status": "success",
                        "id": "post-1",
                        "refId": "ref-1",
                        "profileTitle": "Profile",
                        "post": "hello",
                    }
                ],
            }
        )
    )
    client = AyrshareClient(custom_requests=requests)

    await client.create_post(
        post="hello",
        platforms=[SocialPlatform.TWITTER],
        profile_key=" profile-key ",
    )

    call = requests.calls[0]
    assert call["headers"]["Profile-Key"] == "profile-key"
    assert "Profile-Key" not in client.headers
