from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute
from redis.exceptions import RedisError

from backend.api.external import rate_limit
from backend.api.external.v1 import routes
from backend.api.external.v1.app import v1_app
from backend.blocks._base import BlockType

ROUTES = [
    ("GET", "/me", "read"),
    ("GET", "/blocks", "read"),
    ("POST", "/blocks/{block_id}/execute", "execution"),
    ("POST", "/graphs", "execution"),
    ("POST", "/graphs/{graph_id}/execute/{graph_version}", "execution"),
    ("GET", "/graphs/{graph_id}/executions/{graph_exec_id}/results", "read"),
    ("GET", "/store/agents", "read"),
    ("GET", "/store/agents/{username}/{agent_name}", "read"),
    ("GET", "/store/creators", "read"),
    ("GET", "/store/creators/{username}", "read"),
    ("GET", "/integrations/providers", "read"),
    ("GET", "/integrations/credentials", "read"),
    ("GET", "/integrations/{provider}/credentials", "read"),
    ("POST", "/integrations/{provider}/oauth/initiate", "execution"),
    ("POST", "/integrations/{provider}/oauth/complete", "execution"),
    ("POST", "/integrations/{provider}/credentials", "execution"),
    ("DELETE", "/integrations/{provider}/credentials/{cred_id}", "execution"),
    ("POST", "/tools/find-agent", "execution"),
    ("POST", "/tools/run-agent", "execution"),
]


async def test_read_cap_is_shared_across_credentials_and_resets_at_reported_time(
    v1_client, v1_counter, v1_clock, v1_generated_keys, v1_key_rows
):
    for index in range(rate_limit.READ_MAX_REQUESTS):
        headers = (
            {"X-API-Key": v1_generated_keys[index].key}
            if index < 2
            else {"Authorization": "Bearer agpt_xt_test-0"}
        )
        response = await v1_client.get("/v1/me", headers=headers)
        assert response.status_code == 200
        assert int(response.headers["x-ratelimit-remaining"]) == 299 - index

    headers = {"Authorization": "Bearer agpt_xt_test-1"}
    rejected = await v1_client.get("/v1/me", headers=headers)
    assert rejected.status_code == 429
    assert rejected.headers["x-ratelimit-limit"] == "300"
    assert rejected.headers["x-ratelimit-remaining"] == "0"
    assert rejected.headers["retry-after"] == "3"
    assert "no-store" in rejected.headers["cache-control"]
    assert (
        int(rejected.headers["x-ratelimit-reset"])
        == int(v1_clock.now.return_value.timestamp()) + 3
    )
    assert len({call.args[0] for call in v1_counter.await_args_list}) == 1

    other = await v1_client.get(
        "/v1/me", headers={"Authorization": "Bearer agpt_xt_test-2"}
    )
    assert other.status_code == 200
    assert other.json()["id"] == v1_key_rows[2].user_id
    assert other.headers["x-ratelimit-remaining"] == "299"

    v1_clock.now.return_value += timedelta(seconds=3)
    retried = await v1_client.get("/v1/me", headers=headers)
    assert retried.status_code == 200
    assert retried.headers["x-ratelimit-remaining"] == "299"


async def test_execution_cap_blocks_work_without_consuming_the_read_budget(
    mocker, v1_client, v1_counter
):
    async def execute(*args, **kwargs):
        yield "result", "ok"

    block = MagicMock(
        id="test-block", block_type=BlockType.STANDARD, disabled=False, execute=execute
    )
    get_block = mocker.patch("backend.blocks.get_block", return_value=block)
    charge = mocker.patch.object(routes, "charge_for_direct_block_execution")
    headers = {"Authorization": "Bearer agpt_xt_test-0"}
    for index in range(rate_limit.EXECUTION_MAX_REQUESTS):
        response = await v1_client.post(
            "/v1/blocks/test-block/execute", headers=headers, json={}
        )
        assert response.status_code == 200
        assert response.json() == {"result": ["ok"]}
        assert int(response.headers["x-ratelimit-remaining"]) == 29 - index

    rejected = await v1_client.post(
        "/v1/blocks/test-block/execute", headers=headers, json={}
    )
    assert rejected.status_code == 429
    assert rejected.headers["x-ratelimit-limit"] == "30"
    assert get_block.call_count == charge.await_count == 30
    read = await v1_client.get("/v1/me", headers=headers)
    assert read.status_code == 200
    assert read.headers["x-ratelimit-remaining"] == "299"


def test_route_matrix_covers_every_v1_operation():
    assert {(method, path) for method, path, _ in ROUTES} == {
        (method, route.path)
        for route in v1_app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }


@pytest.mark.parametrize("method,path,tier", ROUTES)
async def test_every_route_enforces_exactly_its_declared_tier(
    mocker, v1_client, v1_counter, v1_key_rows, method, path, tier
):
    limit = 300 if tier == "read" else 30
    v1_counter.side_effect = None
    v1_counter.return_value = limit + 1
    path = path.format(
        block_id="block",
        graph_id="graph",
        graph_version=1,
        graph_exec_id="run",
        username="user",
        agent_name="agent",
        provider="github",
        cred_id="cred",
    )
    metric = mocker.patch.object(rate_limit, "record_rate_limit_hit")
    response = await v1_client.request(
        method,
        f"/v1{path}",
        json={},
        headers={"Authorization": "Bearer agpt_xt_test-0"},
    )

    assert response.status_code == 429
    assert response.headers["x-ratelimit-limit"] == str(limit)
    assert response.headers["x-ratelimit-remaining"] == "0"
    v1_counter.assert_awaited_once()
    assert f":{v1_key_rows[0].user_id}:{tier}:" in v1_counter.await_args.args[0]
    metric.assert_called_once_with(f"external_api:{tier}", v1_key_rows[0].user_id)


async def test_redis_outage_preserves_authentication_and_permissions(
    v1_client, v1_counter, v1_tokens
):
    v1_counter.side_effect = RedisError("unavailable")
    headers = {"Authorization": "Bearer agpt_xt_test-0"}
    allowed = await v1_client.get("/v1/me", headers=headers)
    assert allowed.status_code == 200
    assert "x-ratelimit-limit" not in allowed.headers
    v1_tokens["agpt_xt_test-0"].scopes = []
    assert (await v1_client.get("/v1/me", headers=headers)).status_code == 403
    v1_counter.reset_mock()
    assert (await v1_client.get("/v1/me")).status_code == 401
    v1_counter.assert_not_awaited()
