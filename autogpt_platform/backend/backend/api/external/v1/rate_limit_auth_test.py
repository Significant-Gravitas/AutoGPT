from datetime import UTC, datetime, timedelta
from threading import get_ident

import pytest
from prisma.enums import APIKeyPermission

from backend.api.external import middleware
from backend.data.auth import api_key


@pytest.mark.parametrize("bearer", [False, True], ids=["api-key", "bearer-key"])
async def test_real_key_verification_runs_off_the_route_event_loop(
    mocker, v1_client, v1_counter, v1_generated_keys, v1_key_rows, bearer
):
    verification_threads: list[int] = []
    verify = api_key.keysmith.verify_key

    def traced_verify(*args, **kwargs):
        verification_threads.append(get_ident())
        return verify(*args, **kwargs)

    mocker.patch.object(api_key.keysmith, "verify_key", side_effect=traced_verify)
    key = v1_generated_keys[0].key
    headers = {"Authorization": f"Bearer {key}"} if bearer else {"X-API-Key": key}
    response = await v1_client.get("/v1/me", headers=headers)

    assert response.status_code == 200
    assert response.json()["id"] == v1_key_rows[0].user_id
    assert verification_threads
    assert get_ident() not in verification_threads
    v1_counter.assert_awaited_once()
    assert "no-store" in response.headers["cache-control"]


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-API-Key": "invalid"}, {"Authorization": "Bearer agpt_xt_invalid"}],
    ids=["missing", "invalid-key", "invalid-token"],
)
async def test_authentication_precedes_the_counter(v1_client, v1_counter, headers):
    response = await v1_client.get("/v1/me", headers=headers)

    assert response.status_code == 401
    v1_counter.assert_not_awaited()
    assert "x-ratelimit-limit" not in response.headers


async def test_wrong_key_with_a_valid_head_is_rejected(
    v1_client, v1_counter, v1_generated_keys
):
    key = v1_generated_keys[0].key
    wrong = key[:-1] + ("a" if key[-1] != "a" else "b")

    response = await v1_client.get("/v1/me", headers={"X-API-Key": wrong})

    assert response.status_code == 401
    v1_counter.assert_not_awaited()


async def test_invalid_key_does_not_fall_back_to_a_valid_bearer(
    v1_client, v1_counter, v1_auth_storage
):
    response = await v1_client.get(
        "/v1/me",
        headers={"X-API-Key": "invalid", "Authorization": "Bearer agpt_xt_test-0"},
    )

    assert response.status_code == 401
    v1_counter.assert_not_awaited()
    v1_auth_storage[1].assert_not_awaited()


@pytest.mark.parametrize("credential", ["key", "token"])
async def test_banned_owner_is_rejected_before_the_counter(
    mocker, v1_client, v1_counter, v1_generated_keys, credential
):
    mocker.patch.object(middleware, "is_user_banned", return_value=True)
    headers = (
        {"X-API-Key": v1_generated_keys[0].key}
        if credential == "key"
        else {"Authorization": "Bearer agpt_xt_test-0"}
    )

    response = await v1_client.get("/v1/me", headers=headers)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "account_banned"
    v1_counter.assert_not_awaited()


@pytest.mark.parametrize("state", ["expired", "revoked", "disabled-client"])
async def test_invalid_oauth_state_precedes_the_counter(
    v1_client, v1_counter, v1_tokens, state
):
    token = v1_tokens["agpt_xt_test-0"]
    if state == "expired":
        token.expiresAt = datetime.now(UTC) - timedelta(hours=1)
    elif state == "revoked":
        token.revokedAt = datetime.now(UTC)
    else:
        token.Application.isActive = False

    response = await v1_client.get(
        "/v1/me", headers={"Authorization": "Bearer agpt_xt_test-0"}
    )

    assert response.status_code == 401
    v1_counter.assert_not_awaited()


@pytest.mark.parametrize("path", ["/me", "/blocks", "/integrations/credentials"])
async def test_missing_permissions_never_reach_the_handler(
    mocker, v1_client, v1_counter, v1_tokens, path
):
    v1_tokens["agpt_xt_test-0"].scopes = []
    user = mocker.patch("backend.api.external.v1.routes.user_db.get_user_by_id")
    blocks = mocker.patch("backend.blocks.get_blocks")
    credentials = mocker.patch(
        "backend.api.external.v1.integrations.creds_manager.store.get_all_creds"
    )

    response = await v1_client.get(
        f"/v1{path}", headers={"Authorization": "Bearer agpt_xt_test-0"}
    )

    assert response.status_code == 403
    assert "Missing required permission" in response.json()["detail"]
    user.assert_not_called()
    blocks.assert_not_called()
    credentials.assert_not_called()


async def test_verification_and_permissions_are_not_cached_across_requests(
    v1_client, v1_counter, v1_tokens
):
    headers = {"Authorization": "Bearer agpt_xt_test-0"}
    assert (await v1_client.get("/v1/me", headers=headers)).status_code == 200
    v1_tokens["agpt_xt_test-0"].scopes.remove(APIKeyPermission.IDENTITY)
    assert (await v1_client.get("/v1/me", headers=headers)).status_code == 403
    v1_tokens["agpt_xt_test-0"].revokedAt = datetime.now(UTC)
    assert (await v1_client.get("/v1/me", headers=headers)).status_code == 401
