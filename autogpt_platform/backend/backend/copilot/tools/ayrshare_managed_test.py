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
from backend.copilot.context import set_turn_unattended
from backend.copilot.tools.connect_integration import ConnectIntegrationTool
from backend.copilot.tools.helpers import execute_block, prepare_block_for_execution
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    SetupRequirementsResponse,
)
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


async def _prepare(block_id: str, *, dry_run: bool = False):
    return await prepare_block_for_execution(
        block_id=block_id,
        input_data={"post": "Hello from AutoPilot"},
        user_id=USER,
        session=make_session(USER),
        session_id="s-ayrshare",
        dry_run=dry_run,
    )


@pytest.mark.asyncio(loop_scope="session")
class TestBlockWithoutCredential:
    @pytest.mark.parametrize("block", [PostToLinkedInBlock(), PostToXBlock()])
    async def test_provisions_the_profile_and_returns_the_link(self, block):
        # Repro of SECRT-2878: every prod session ended on "requires
        # credentials that are not configured".
        with ayrshare_backend() as provision:
            result = await _prepare(block.id)

        assert isinstance(result, SetupRequirementsResponse)
        assert "not configured" not in result.message
        assert SSO_URL in result.message
        provision.assert_awaited_once()
        (entry,) = result.setup_info.user_readiness.missing_credentials.values()
        assert entry["provider"] == "ayrshare"
        assert entry["connect_url"] == SSO_URL

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

    async def test_dry_run_provisions_nothing(self):
        with ayrshare_backend() as provision:
            await _prepare(PostToLinkedInBlock().id, dry_run=True)

        provision.assert_not_awaited()

    async def test_unattended_turn_gets_the_link_in_an_error(self):
        set_turn_unattended(True)
        try:
            with ayrshare_backend():
                result = await _prepare(PostToLinkedInBlock().id)
        finally:
            set_turn_unattended(False)

        assert isinstance(result, ErrorResponse)
        assert result.error == "ayrshare_accounts_not_linked"
        assert SSO_URL in result.message


@pytest.mark.asyncio(loop_scope="session")
class TestConnectIntegration:
    async def test_ayrshare_returns_the_managed_link(self):
        with ayrshare_backend() as provision:
            result = await ConnectIntegrationTool()._execute(
                USER, make_session(USER), provider="ayrshare"
            )

        assert isinstance(result, SetupRequirementsResponse)
        assert SSO_URL in result.message
        provision.assert_awaited_once()
        entry = result.setup_info.user_readiness.missing_credentials[
            "ayrshare_credentials"
        ]
        assert entry["provider"] == "ayrshare"
        assert entry["connect_url"] == SSO_URL

    async def test_ayrshare_not_configured(self):
        with ayrshare_backend(configured=False):
            result = await ConnectIntegrationTool()._execute(
                USER, make_session(USER), provider="ayrshare"
            )

        assert isinstance(result, ErrorResponse)
        assert result.error == "ayrshare_not_configured"

    def test_ayrshare_is_offered(self):
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
    async def test_returns_the_link_and_charges_nothing(self):
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
        assert SSO_URL in result.message
        assert "Twitter is not linked" in result.message
        spend.assert_not_awaited()

    async def test_other_ayrshare_errors_stay_errors(self):
        post = AsyncMock(
            side_effect=BlockExecutionError("Duplicate post", "PostToXBlock", "x")
        )
        result = await _run(PostToXBlock(), post, AsyncMock())

        assert isinstance(result, ErrorResponse)
        assert "Duplicate post" in result.message
