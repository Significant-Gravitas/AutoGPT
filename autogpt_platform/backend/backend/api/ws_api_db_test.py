"""The websocket server run as a deployment runs it, with no database of its
own: its reads must reach the DatabaseManager (AUTOGPT-SERVER-8VJ)."""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import prisma.models
import pytest

from backend.api.conn_manager import graph_exec_channel_key
from backend.api.model import WSMethod
from backend.util.clients import get_database_manager_async_client
from backend.util.test import SpinTestServer

# Nothing listens on port 1, so a Prisma client in the probe can never connect.
_UNREACHABLE_DATABASE = "postgresql://nobody:nothing@127.0.0.1:1/none"

_PROBE = """
import json
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.api import ws_api
from backend.api.model import WSMessage, WSMethod

user_id, banned_user_id, graph_exec_id = sys.argv[1:4]


async def token_subject(token):
    return {"sub": token}


subscribe = WSMessage(
    method=WSMethod.SUBSCRIBE_GRAPH_EXEC, data={"graph_exec_id": graph_exec_id}
).model_dump_json()
with (
    patch.object(ws_api, "parse_jwt_token_async", token_subject),
    TestClient(ws_api.app) as client,
):
    with client.websocket_connect(f"/ws?token={user_id}") as socket:
        socket.send_text(subscribe)
        reply = json.loads(socket.receive_text())
    try:
        with client.websocket_connect(f"/ws?token={banned_user_id}"):
            banned_close = None
    except WebSocketDisconnect as refusal:
        banned_close = refusal.code
print(json.dumps({"reply": reply, "banned_close": banned_close}))
"""


@pytest.mark.asyncio(loop_scope="session")
async def test_the_server_answers_sockets_with_no_database_of_its_own(
    server: SpinTestServer,
):
    user_id, graph_id, graph_exec_id = await _seed_execution()
    banned_user_id = await _seed_banned_user()
    # The probe's reads land here; a cold server is not what is measured.
    await get_database_manager_async_client().health_check_async()
    try:
        outcome = await _run_probe(user_id, banned_user_id, graph_exec_id)
    finally:
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})
        await prisma.models.User.prisma().delete_many(where={"id": user_id})
        await prisma.models.AuthUser.prisma().delete_many(where={"id": banned_user_id})

    assert outcome["reply"] == {
        "method": WSMethod.SUBSCRIBE_GRAPH_EXEC.value,
        "data": None,
        "success": True,
        "channel": graph_exec_channel_key(user_id, graph_exec_id=graph_exec_id),
        "error": None,
    }
    assert outcome["banned_close"] == 4004


async def _seed_execution() -> tuple[str, str, str]:
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"ws-probe-{user_id}@example.com"}
    )
    await prisma.models.AgentGraph.prisma().create(
        data={"id": graph_id, "version": 1, "name": "ws-probe", "userId": user_id}
    )
    execution = await prisma.models.AgentGraphExecution.prisma().create(
        data={"agentGraphId": graph_id, "agentGraphVersion": 1, "userId": user_id}
    )
    return user_id, graph_id, execution.id


async def _seed_banned_user() -> str:
    user_id = str(uuid.uuid4())
    await prisma.models.AuthUser.prisma().create(
        data={
            "id": user_id,
            "name": "ws-probe",
            "email": f"ws-probe-banned-{user_id}@example.com",
            "banned": True,
        }
    )
    return user_id


async def _run_probe(user_id: str, banned_user_id: str, graph_exec_id: str) -> dict:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _PROBE,
        user_id,
        banned_user_id,
        graph_exec_id,
        cwd=Path(__file__).parents[2],
        env={
            **os.environ,
            "DATABASE_URL": _UNREACHABLE_DATABASE,
            "DIRECT_URL": _UNREACHABLE_DATABASE,
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    assert process.returncode == 0, stderr.decode()[-4000:]
    return json.loads(stdout.decode().strip().splitlines()[-1])
