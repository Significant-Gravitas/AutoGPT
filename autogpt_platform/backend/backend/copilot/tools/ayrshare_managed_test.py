"""AutoPilot and the managed Ayrshare profile (SECRT-2878).

The repro tests patch only names that exist before the fix too (the org
secrets, the provisioning lock, the credential store and Ayrshare's JWT
call), so on the old code they fail on behaviour, not on a missing symbol.
"""

from contextlib import ExitStack, contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.ayrshare.post_to_linkedin import PostToLinkedInBlock
from backend.blocks.ayrshare.post_to_x import PostToXBlock
from backend.copilot.context import _current_permissions, set_turn_unattended
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools.ayrshare_managed import (
    BUILDER_CONNECT_BUTTON,
    GRANT_TO_EXPERT,
)
from backend.copilot.tools.connect_integration import ConnectIntegrationTool
from backend.copilot.tools.helpers import execute_block, prepare_block_for_execution
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    SetupRequirementsResponse,
    ToolResponseBase,
)
from backend.copilot.tools.run_block import RunBlockTool
from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.integrations.ayrshare import (
    AyrshareClient,
    JWTResponse,
    PostIds,
    PostResponse,
)
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.integrations.providers import ProviderName
from backend.util.exceptions import BlockExecutionError

from ._test_data import make_session

USER = "user-ayrshare-copilot"
SSO_URL = "https://profile.ayrshare.com/social-accounts?domain=id-pojeg&jwt=tok"
MANAGED = APIKeyCredentials(
    id="managed-ayrshare",
    provider="ayrshare",
    title="Ayrshare (managed by AutoGPT)",
    api_key=SecretStr("profile-key"),
    expires_at=None,
    is_managed=True,
)


@contextmanager
def ayrshare_backend(
    *, configured: bool = True, has_profile: bool = False
) -> Iterator[MagicMock]:
    """Ayrshare as the deployment sees it: org secrets, the user's profile
    and the JWT endpoint. Yields the provisioning mock."""
    secrets = MagicMock()
    secrets.secrets.ayrshare_api_key = "org-key" if configured else ""
    secrets.secrets.ayrshare_jwt_key = "jwt-key" if configured else ""
    provision = AsyncMock(return_value=True)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "backend.integrations.managed_providers.ayrshare.Settings",
                return_value=secrets,
            )
        )
        stack.enter_context(patch("backend.integrations.ayrshare.settings", secrets))
        stack.enter_context(
            patch(
                "backend.integrations.managed_credentials._provision_under_lock",
                provision,
            )
        )
        stack.enter_context(
            patch.object(
                IntegrationCredentialsStore,
                "has_managed_credential",
                AsyncMock(return_value=has_profile),
            )
        )
        stack.enter_context(
            patch.object(
                IntegrationCredentialsStore,
                "get_creds_by_provider",
                AsyncMock(return_value=[MANAGED]),
            )
        )
        stack.enter_context(
            patch.object(
                AyrshareClient,
                "generate_jwt",
                AsyncMock(
                    return_value=JWTResponse(
                        status="success", title="t", token="tok", url=SSO_URL
                    )
                ),
            )
        )
        stack.enter_context(
            patch(
                "backend.copilot.tools.utils.get_user_credentials",
                AsyncMock(return_value=[]),
            )
        )
        stack.enter_context(
            patch(
                "backend.copilot.tools.utils.selected_credentials",
                AsyncMock(return_value={}),
            )
        )
        yield provision


async def _prepare(block_id: str, *, session=None):
    return await prepare_block_for_execution(
        block_id=block_id,
        input_data={"post": "Hello from AutoPilot"},
        user_id=USER,
        session=session or make_session(USER),
        session_id="s-ayrshare",
        dry_run=False,
    )


def assert_no_sso_link(result: ToolResponseBase) -> None:
    # Tool output is stored with the chat, sent to the model provider and
    # served by a shared chat; the link signs into the user's Ayrshare page.
    dumped = result.model_dump_json()
    assert SSO_URL not in dumped
    assert "jwt=" not in dumped
    assert "connect_url" not in dumped


@pytest.mark.asyncio(loop_scope="session")
class TestBlockWithoutCredential:
    @pytest.mark.parametrize("block", [PostToLinkedInBlock(), PostToXBlock()])
    async def test_provisions_the_profile_and_returns_the_connect_card(self, block):
        # Repro of SECRT-2878: every prod session ended on "requires
        # credentials that are not configured".
        with ayrshare_backend() as provision:
            result = await _prepare(block.id)

        assert isinstance(result, SetupRequirementsResponse)
        assert "not configured" not in result.message
        # The chat card's Ayrshare row reads "Connect"; the builder's longer
        # label is not on the card, so the model must not point at it.
        assert "with the Connect button in the card" in result.message
        assert BUILDER_CONNECT_BUTTON not in result.message
        assert "Each post costs $0.02 ($0.05 for video)." in result.message
        provision.assert_awaited_once()
        (entry,) = result.setup_info.user_readiness.missing_credentials.values()
        assert entry["provider"] == "ayrshare"

    async def test_not_configured_says_it_is_unavailable(self):
        with ayrshare_backend(configured=False) as provision:
            result = await _prepare(PostToLinkedInBlock().id)

        assert isinstance(result, ErrorResponse)
        assert result.error == "ayrshare_not_configured"
        assert "not available" in result.message
        provision.assert_not_awaited()

    async def test_existing_profile_kept_out_of_the_match_is_not_reprovisioned(self):
        # An expert grant or a pinned account hid it: the usual card handles it.
        with ayrshare_backend(has_profile=True) as provision:
            result = await _prepare(PostToLinkedInBlock().id)

        assert isinstance(result, SetupRequirementsResponse)
        assert "not configured" in result.message
        provision.assert_not_awaited()

    @pytest.mark.parametrize("dry_run,validate_only", [(True, False), (False, True)])
    async def test_inspection_provisions_nothing(self, dry_run, validate_only):
        with ayrshare_backend() as provision:
            await prepare_block_for_execution(
                block_id=PostToLinkedInBlock().id,
                input_data={"post": "Hello from AutoPilot"},
                user_id=USER,
                session=make_session(USER),
                session_id="s-ayrshare",
                dry_run=dry_run,
                validate_only=validate_only,
            )

        provision.assert_not_awaited()

    async def test_unattended_turn_says_where_to_connect(self):
        session = make_session(USER)
        set_turn_unattended(session, scheduled=True)
        try:
            with ayrshare_backend():
                result = await _prepare(PostToLinkedInBlock().id)
        finally:
            set_turn_unattended(session, scheduled=False)

        assert isinstance(result, ErrorResponse)
        assert result.error == "ayrshare_accounts_not_linked"
        assert "choosing Connect on the Ayrshare card" in result.message
        assert f"or {BUILDER_CONNECT_BUTTON} in the workflow builder" in result.message

    async def test_a_block_the_session_may_not_run_provisions_nothing(self):
        block_id = PostToLinkedInBlock().id
        token = _current_permissions.set(
            CopilotPermissions(blocks=[block_id], blocks_exclude=True)
        )
        try:
            with ayrshare_backend() as provision:
                result = await RunBlockTool()._execute(
                    USER,
                    make_session(USER),
                    block_id=block_id,
                    input_data={"post": "Hello from AutoPilot"},
                )
        finally:
            _current_permissions.reset(token)

        assert isinstance(result, ErrorResponse)
        assert "not permitted" in result.message
        assert SSO_URL not in result.message
        provision.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
class TestConnectIntegration:
    async def test_ayrshare_returns_the_managed_connect_card(self):
        with ayrshare_backend() as provision:
            result = await ConnectIntegrationTool()._execute(
                USER, make_session(USER), provider="ayrshare"
            )

        assert isinstance(result, SetupRequirementsResponse)
        assert "with the Connect button in the card" in result.message
        provision.assert_awaited_once()
        entry = result.setup_info.user_readiness.missing_credentials[
            "ayrshare_credentials"
        ]
        assert entry["provider"] == "ayrshare"

    async def test_ayrshare_not_configured(self):
        with ayrshare_backend(configured=False):
            result = await ConnectIntegrationTool()._execute(
                USER, make_session(USER), provider="ayrshare"
            )

        assert isinstance(result, ErrorResponse)
        assert result.error == "ayrshare_not_configured"

    async def test_a_dry_run_creates_nothing(self):
        session = make_session(USER)
        session.metadata.dry_run = True
        with ayrshare_backend() as provision:
            result = await ConnectIntegrationTool()._execute(
                USER, session, provider="ayrshare"
            )

        assert isinstance(result, ErrorResponse)
        assert result.error == "dry_run"
        assert "nothing was created" in result.message.lower()
        assert SSO_URL not in result.message
        provision.assert_not_awaited()


def test_connect_integration_offers_ayrshare():
    tool = ConnectIntegrationTool()
    assert tool.parameters["properties"]["provider"]["enum"] == [
        "github",
        "ayrshare",
    ]
    assert "'ayrshare'" in tool.description


def _patch_execution(spend: AsyncMock, *, post: AsyncMock | None = None):
    """The platform around a real block run: workspace, user, credit store,
    the stored managed credential and Ayrshare's post endpoint."""
    workspace = MagicMock()
    workspace.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
    users = MagicMock()
    users.get_user_by_id = AsyncMock(return_value=MagicMock(timezone="UTC"))
    credits = MagicMock(get_credits=AsyncMock(return_value=100), spend_credits=spend)
    creds = MagicMock(get=AsyncMock(return_value=MANAGED))
    secrets = MagicMock()
    secrets.secrets.ayrshare_api_key = "org-key"
    stack = ExitStack()
    for target, value in (
        ("backend.copilot.tools.helpers.workspace_db", workspace),
        ("backend.copilot.tools.helpers.user_db", users),
        ("backend.copilot.tools.helpers.credit_db", credits),
        ("backend.copilot.tools.helpers.IntegrationCredentialsManager", creds),
    ):
        stack.enter_context(patch(target, return_value=value))
    stack.enter_context(patch("backend.integrations.ayrshare.settings", secrets))
    if post is not None:
        stack.enter_context(patch.object(AyrshareClient, "create_post", post))
    return stack


async def _run(block, post: AsyncMock, spend: AsyncMock, **inputs):
    meta = CredentialsMetaInput(
        id=MANAGED.id,
        provider=ProviderName("ayrshare"),
        type="api_key",
    )
    with _patch_execution(spend, post=post):
        return await execute_block(
            block=block,
            block_id=block.id,
            input_data={"post": "Hello", **inputs},
            user_id=USER,
            session_id="s-ayrshare",
            node_exec_id="n-ayrshare",
            matched_credentials={"credentials": meta},
            dry_run=False,
        )


def _posted() -> AsyncMock:
    return AsyncMock(
        return_value=PostResponse(
            status="success",
            id="p-1",
            refId="r-1",
            profileTitle="User",
            post="Hello",
            postIds=[
                PostIds(
                    status="success",
                    id="li-1",
                    postUrl="https://linkedin.com/p/1",
                    platform="linkedin",
                )
            ],
        )
    )


@pytest.mark.asyncio(loop_scope="session")
class TestManagedPostBilling:
    @pytest.mark.parametrize("is_video,cost", [(False, 2), (True, 5)])
    async def test_a_managed_post_charges_the_block_credits(self, is_video, cost):
        spend = AsyncMock()
        post = _posted()
        result = await _run(PostToLinkedInBlock(), post, spend, is_video=is_video)

        assert isinstance(result, BlockOutputResponse)
        assert post.await_args.kwargs["profile_key"] == "profile-key"
        spend.assert_awaited_once()
        assert spend.await_args.kwargs["cost"] == cost


@pytest.mark.asyncio(loop_scope="session")
class TestPostToUnlinkedNetwork:
    async def test_returns_the_connect_card_and_charges_nothing(self):
        spend = AsyncMock()
        post = AsyncMock(
            side_effect=BlockExecutionError(
                "Ayrshare API request failed (400): Twitter is not linked",
                "PostToXBlock",
                "x",
            )
        )
        with ayrshare_backend(has_profile=True):
            result = await _run(PostToXBlock(), post, spend)

        assert isinstance(result, SetupRequirementsResponse)
        assert "with the Connect button in the card" in result.message
        assert "Twitter is not linked" in result.message
        spend.assert_not_awaited()

    async def test_other_ayrshare_errors_stay_errors(self):
        post = AsyncMock(
            side_effect=BlockExecutionError("Duplicate post", "PostToXBlock", "x")
        )
        result = await _run(PostToXBlock(), post, AsyncMock())

        assert isinstance(result, ErrorResponse)
        assert "Duplicate post" in result.message


def _unlinked_post() -> AsyncMock:
    return AsyncMock(
        side_effect=BlockExecutionError(
            "Ayrshare API request failed (400): Twitter is not linked",
            "PostToXBlock",
            "x",
        )
    )


async def _block_without_credential(session):
    return await _prepare(PostToLinkedInBlock().id, session=session)


async def _connect(session):
    return await ConnectIntegrationTool()._execute(USER, session, provider="ayrshare")


async def _post_to_unlinked_network(session):
    return await _run(PostToXBlock(), _unlinked_post(), AsyncMock())


@pytest.mark.asyncio(loop_scope="session")
class TestNoSSOLinkInChat:
    @pytest.mark.parametrize("unattended", [False, True])
    @pytest.mark.parametrize(
        "answer,has_profile",
        [
            (_block_without_credential, False),
            (_connect, False),
            (_connect, True),
            (_post_to_unlinked_network, True),
        ],
    )
    async def test_no_response_carries_the_sso_link(
        self, answer, has_profile, unattended
    ):
        session = make_session(USER)
        set_turn_unattended(session, scheduled=unattended)
        signed = JWTResponse(status="success", title="t", token="tok", url=SSO_URL)
        try:
            with (
                ayrshare_backend(has_profile=has_profile),
                patch.object(
                    AyrshareClient, "generate_jwt", AsyncMock(return_value=signed)
                ) as sign,
            ):
                result = await answer(session)
        finally:
            set_turn_unattended(session, scheduled=False)

        if unattended and answer is not _connect:
            assert isinstance(result, ErrorResponse)
            assert result.error == "ayrshare_accounts_not_linked"
        else:
            assert isinstance(result, SetupRequirementsResponse)
        assert_no_sso_link(result)
        sign.assert_not_awaited()


EXPERT = "expert-ayrshare"


async def _expert_block_without_credential():
    return await _prepare(
        PostToLinkedInBlock().id, session=make_session(USER, expert_id=EXPERT)
    )


async def _expert_connect():
    return await ConnectIntegrationTool()._execute(
        USER, make_session(USER, expert_id=EXPERT), provider="ayrshare"
    )


@contextmanager
def expert_grants(allowed: list[str]) -> Iterator[None]:
    """An expert whose grant list is *allowed*, on an account whose only
    credential is the managed profile the copilot just provisioned."""
    experts = MagicMock()
    experts.expert_allowed_credential_ids = AsyncMock(return_value=allowed)
    with (
        patch("backend.copilot.tools.expert_scope.experts_db", return_value=experts),
        patch.object(
            IntegrationCredentialsStore,
            "get_all_creds",
            AsyncMock(return_value=[MANAGED]),
        ),
    ):
        yield


@pytest.mark.asyncio(loop_scope="session")
class TestExpertSession:
    # An expert runs only with credentials it was granted, the managed one
    # included, so the card must offer the grant or linking alone leaves the
    # expert unable to post.
    @pytest.mark.parametrize(
        "answer", [_expert_block_without_credential, _expert_connect]
    )
    async def test_the_card_offers_the_profile_to_the_expert(self, answer):
        with ayrshare_backend(), expert_grants([]):
            result = await answer()

        assert isinstance(result, SetupRequirementsResponse)
        (entry,) = result.setup_info.user_readiness.missing_credentials.values()
        assert entry["expert_grant"]["expert_id"] == EXPERT
        assert [c["id"] for c in entry["expert_grant"]["credentials"]] == [MANAGED.id]
        assert GRANT_TO_EXPERT in result.message
        assert_no_sso_link(result)

    @pytest.mark.parametrize(
        "answer", [_expert_block_without_credential, _expert_connect]
    )
    async def test_an_expert_already_granted_is_not_asked_again(self, answer):
        with ayrshare_backend(), expert_grants([MANAGED.id]):
            result = await answer()

        assert isinstance(result, SetupRequirementsResponse)
        (entry,) = result.setup_info.user_readiness.missing_credentials.values()
        assert entry["expert_grant"]["credentials"] == []
        assert GRANT_TO_EXPERT not in result.message

    async def test_unattended_says_to_grant_the_expert(self):
        session = make_session(USER, expert_id=EXPERT)
        set_turn_unattended(session, scheduled=True)
        try:
            with ayrshare_backend(), expert_grants([]):
                result = await _prepare(PostToLinkedInBlock().id, session=session)
        finally:
            set_turn_unattended(session, scheduled=False)

        assert isinstance(result, ErrorResponse)
        assert result.error == "ayrshare_accounts_not_linked"
        assert "grant this expert access" in result.message
        assert_no_sso_link(result)
