"""A block run directly runs on the caller's own stored credentials, and a
request naming credentials it can't run on is refused before it costs anything.

Through both block-execute routes, on the test stack: Postgres holds the users,
the credentials they stored, their credit balances and a v1 API key, and each
block runs on a direct-execution worker, which loads its credentials through the
database manager as the graph executor does. The blocks are test blocks that
report the credentials they ran with, priced like a block on the platform's key.
Nothing of the credential layer is stubbed; only the internal route's sign-in
token is, as in the other route tests.
"""

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth.dependencies import get_unbanned_jwt_payload, get_user_id
from fastapi import Request
from prisma.enums import APIKeyPermission, CreditTransactionType
from prisma.models import CreditTransaction, Organization, User, UserBalance
from pydantic import BaseModel, SecretStr
from pytest_mock import MockerFixture

from backend.api.external.fastapi_app import external_api
from backend.api.rest_api import app as rest_app
from backend.blocks import get_blocks
from backend.blocks._base import (
    Block,
    BlockCost,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data import credit
from backend.data.auth import api_key
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
)
from backend.data.org_migration import create_personal_org
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName

PRICE = 7
OWNERS_SECRET = "sk-owner-direct-run"
NEEDS_AN_OPENAI_KEY = (
    "needs credentials with provider 'openai' and type 'api_key' in its "
    "'credentials' input."
)

# What each test block ran with, appended on the worker that ran it.
RUNS: list[str] = []
LEASE_BLOCK_STARTED = threading.Event()
LEASE_BLOCK_MAY_FINISH = threading.Event()


class CredentialsEchoBlock(Block):
    """Answers with the OpenAI key its run was handed, as a credentialed
    block's ``run`` takes it (``IdeogramModelBlock`` in #17321)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.OPENAI], Literal["api_key"]
        ] = CredentialsField(description="An OpenAI API key")

    class Output(BlockSchemaOutput):
        credentials_id: str = SchemaField(description="The credentials' id")
        api_key: str = SchemaField(description="The key the block ran with")

    def __init__(self):
        super().__init__(
            id="5c4b1f7e-3a2d-4e8f-9b6a-0d1c2e3f4a5b",
            description="Reports the OpenAI key it runs with.",
            input_schema=CredentialsEchoBlock.Input,
            output_schema=CredentialsEchoBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        RUNS.append(credentials.id)
        yield "credentials_id", credentials.id
        yield "api_key", credentials.api_key.get_secret_value()


class OAuthLeaseBlock(Block):
    """Runs on a GitHub OAuth token until the test lets it finish, then
    succeeds or fails as its input says."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.GITHUB], Literal["oauth2"]
        ] = CredentialsField(description="A GitHub account")
        fail: bool = SchemaField(default=False, description="Fail once released")

    class Output(BlockSchemaOutput):
        credentials_id: str = SchemaField(description="The credentials' id")

    def __init__(self):
        super().__init__(
            id="8e7d6c5b-4a3f-4b2e-9d1c-0f9e8d7c6b5a",
            description="Holds its run until released.",
            input_schema=OAuthLeaseBlock.Input,
            output_schema=OAuthLeaseBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
    ) -> BlockOutput:
        RUNS.append(credentials.id)
        LEASE_BLOCK_STARTED.set()
        await asyncio.to_thread(LEASE_BLOCK_MAY_FINISH.wait, 30)
        if input_data.fail:
            raise ValueError("The test block failed after it started.")
        yield "credentials_id", credentials.id


ECHO = CredentialsEchoBlock().id
LEASE = OAuthLeaseBlock().id


class World(BaseModel):
    """``owner`` keeps an OpenAI key, an Anthropic key, a GitHub account, credits
    and a v1 key that may run blocks; ``stranger`` keeps an OpenAI key."""

    owner: str
    stranger: str
    v1_key: str
    openai_key: str
    anthropic_key: str
    github_account: str
    strangers_key: str


RunBlock = Callable[[str, dict[str, Any]], Awaitable[httpx.Response]]


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def world(server) -> AsyncIterator[World]:
    owner, stranger = (f"direct-run-{uuid.uuid4()}" for _ in range(2))
    organizations: list[str] = []
    try:
        for user_id in (owner, stranger):
            await User.prisma().create(
                data={"id": user_id, "email": f"{user_id}@example.com"}
            )
            organizations.append(
                (await create_personal_org(user_id, user_id, user_id)).id
            )
        await credit.UserCredit()._add_transaction(
            owner, amount=100 * PRICE, transaction_type=CreditTransactionType.GRANT
        )
        _, v1_key = await api_key.create_api_key(
            name="direct run",
            user_id=owner,
            permissions=[APIKeyPermission.EXECUTE_BLOCK],
        )
        yield World(
            owner=owner,
            stranger=stranger,
            v1_key=v1_key,
            openai_key=await _store(owner, _api_key("openai", OWNERS_SECRET)),
            anthropic_key=await _store(owner, _api_key("anthropic", "sk-ant-owner")),
            github_account=await _store(
                owner,
                OAuth2Credentials(
                    provider="github",
                    title="Owner's GitHub",
                    access_token=SecretStr("gho-owner"),
                    scopes=["repo"],
                ),
            ),
            strangers_key=await _store(stranger, _api_key("openai", "sk-stranger")),
        )
    finally:
        users = {"in": [owner, stranger]}
        await CreditTransaction.prisma().delete_many(where={"userId": users})
        await UserBalance.prisma().delete_many(where={"userId": users})
        await User.prisma().delete_many(where={"id": users})
        await Organization.prisma().delete_many(where={"id": {"in": organizations}})


@pytest.fixture(autouse=True)
def test_blocks_on_the_platform(mocker: MockerFixture) -> None:
    """The test blocks are in the block registry, the echo block costs ``PRICE``
    a run whatever its credentials, and credits are on."""
    mocker.patch.dict(
        get_blocks(), {ECHO: CredentialsEchoBlock, LEASE: OAuthLeaseBlock}
    )
    mocker.patch.dict(BLOCK_COSTS, {CredentialsEchoBlock: [BlockCost(PRICE)]})
    mocker.patch.object(credit.settings.config, "enable_credit", True)
    RUNS.clear()
    LEASE_BLOCK_STARTED.clear()
    LEASE_BLOCK_MAY_FINISH.clear()


@pytest_asyncio.fixture(loop_scope="session", params=["internal", "v1"])
async def run_block(
    request, world: World, mocker: MockerFixture
) -> AsyncIterator[RunBlock]:
    """Runs a block for ``world.owner`` through one of the two routes: the web
    app's, signed in, or v1's, with their API key."""
    if request.param == "internal":
        app, path, headers = rest_app, "/api/blocks/{}/execute", {}

        async def signed_in(_: Request) -> dict[str, str]:
            return {"sub": world.owner, "role": "authenticated", "email": "o@x.io"}

        async def owner() -> str:
            return world.owner

        # The test server signs every request in as its default user through
        # `get_user_id` (SpinTestServer); this test's caller is the owner.
        mocker.patch.dict(
            rest_app.dependency_overrides,
            {get_unbanned_jwt_payload: signed_in, get_user_id: owner},
        )
    else:
        app, path = external_api, "/v1/blocks/{}/execute"
        headers = {"X-API-Key": world.v1_key}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:

        async def run(block_id: str, body: dict[str, Any]) -> httpx.Response:
            return await client.post(path.format(block_id), json=body, headers=headers)

        yield run


async def test_a_block_runs_on_the_callers_stored_credentials(
    world: World, run_block: RunBlock
):
    before = await _balance(world.owner)

    response = await run_block(
        ECHO, {"credentials": _naming(world.openai_key, "openai")}
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "credentials_id": [world.openai_key],
        "api_key": [OWNERS_SECRET],
    }
    assert RUNS == [world.openai_key]
    assert await _balance(world.owner) == before - PRICE


REFUSED_CREDENTIALS = {
    "another user's credentials": lambda world: {
        "credentials": _naming(world.strangers_key, "openai")
    },
    "credentials that don't exist": lambda world: {
        "credentials": _naming(str(uuid.uuid4()), "openai")
    },
    "credentials of another provider, named as this one's": lambda world: {
        "credentials": _naming(world.anthropic_key, "openai")
    },
    "credentials of another provider": lambda world: {
        "credentials": _naming(world.anthropic_key, "anthropic")
    },
    "no credentials": lambda world: {},
}


@pytest.mark.parametrize("case", REFUSED_CREDENTIALS)
async def test_credentials_the_block_cant_run_on_are_refused_before_any_charge(
    world: World, run_block: RunBlock, case: str
):
    before = await _balance(world.owner)

    response = await run_block(ECHO, REFUSED_CREDENTIALS[case](world))

    assert response.status_code == 400, response.text
    assert NEEDS_AN_OPENAI_KEY in response.json()["detail"]
    assert await _balance(world.owner) == before
    assert RUNS == []


async def test_another_users_credentials_are_refused_like_ones_that_dont_exist(
    world: World, run_block: RunBlock
):
    """Nothing in the refusal tells the caller someone else holds the id."""
    unknown = str(uuid.uuid4())

    theirs = await run_block(
        ECHO, {"credentials": _naming(world.strangers_key, "openai")}
    )
    nobodys = await run_block(ECHO, {"credentials": _naming(unknown, "openai")})

    assert theirs.status_code == nobodys.status_code == 400
    assert theirs.json()["detail"].replace(world.strangers_key, "#") == (
        nobodys.json()["detail"].replace(unknown, "#")
    )


@pytest.mark.parametrize("fails", [False, True], ids=["succeeds", "fails"])
async def test_an_oauth_lease_is_held_while_the_block_runs_and_released_after(
    world: World, run_block: RunBlock, fails: bool
):
    manager = IntegrationCredentialsManager()
    run = asyncio.ensure_future(
        run_block(
            LEASE,
            {"credentials": _naming(world.github_account, "github"), "fail": fails},
        )
    )
    try:
        assert await asyncio.to_thread(LEASE_BLOCK_STARTED.wait, 30)
        # The run holds the lease, so another can't take it meanwhile.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                manager.acquire_lease(world.owner, world.github_account), timeout=1
            )
    finally:
        LEASE_BLOCK_MAY_FINISH.set()
    response = await run

    assert response.status_code == (400 if fails else 200), response.text
    assert RUNS == [world.github_account]
    # Released when the block stopped, not left to its lock's 60 s expiry.
    lease = await asyncio.wait_for(
        manager.acquire_lease(world.owner, world.github_account), timeout=30
    )
    await lease.release()


def _api_key(provider: str, secret: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        provider=provider, title=f"A {provider} key", api_key=SecretStr(secret)
    )


async def _store(user_id: str, credentials: Credentials) -> str:
    await IntegrationCredentialsManager().create(user_id, credentials)
    return credentials.id


def _naming(credentials_id: str, provider: str) -> dict[str, str]:
    """A credentials input naming stored credentials, as a caller sends one."""
    credential_type = "oauth2" if provider == "github" else "api_key"
    return {"id": credentials_id, "provider": provider, "type": credential_type}


async def _balance(user_id: str) -> int:
    return await credit.UserCredit().get_credits(user_id)
