"""Connected-but-short credentials are not "not connected".

GitHub without read:org and Linear without comments:create both used to get
the same "connect your account" card as a user who had never connected, and
the reconnect it started asked only for the missing scope.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.linear.comment import LinearCreateCommentBlock
from backend.data.model import (
    CredentialsFieldInfo,
    CredentialsMetaInput,
    CredentialsType,
    OAuth2Credentials,
)
from backend.integrations.oauth.refresh_failure import (
    RECONNECT_REQUIRED_KEY,
    CredentialsNeedReconnectError,
    reconnect_required,
)
from backend.integrations.providers import ProviderName

from ._test_data import make_session
from .connect_integration import ConnectIntegrationTool
from .credential_gaps import (
    annotate_credential_gaps,
    credential_gap_message,
    find_credential_gap,
)
from .helpers import _build_credential_rejected_card
from .models import SetupRequirementsResponse
from .utils import find_matching_credential

USER = "user-credential-gaps"
DEAD = {RECONNECT_REQUIRED_KEY: {"error_code": "invalid_grant", "status_code": 400}}


def oauth(
    provider: str,
    scopes: list[str],
    *,
    cred_id: str = "cred-1",
    metadata: dict | None = None,
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider=provider,
        title=f"alice's {provider}",
        username="alice",
        access_token=SecretStr("at"),
        refresh_token=SecretStr("rt"),
        scopes=scopes,
        metadata=metadata or {},
    )


def requirement(provider: str, scopes: set[str]) -> CredentialsFieldInfo:
    return CredentialsFieldInfo[ProviderName, CredentialsType](
        credentials_provider=frozenset([ProviderName(provider)]),
        credentials_types=frozenset(["oauth2"]),
        credentials_scopes=frozenset(scopes),
    )


# -- Telling "short of a scope" apart from "not connected" -- #


def test_github_without_read_org_is_a_scope_gap_not_a_missing_account():
    gap = find_credential_gap(
        [oauth("github", ["repo"])], requirement("github", {"repo", "read:org"})
    )

    assert gap is not None
    assert gap.kind == "missing_scopes"
    assert gap.missing_scopes == ["read:org"]
    assert gap.granted_scopes == ["repo"]
    assert credential_gap_message("GitHub", gap) == (
        "Your GitHub account 'alice's github' is connected, but it was not granted "
        "the read:org permission this needs. Reconnect it and approve that "
        "permission."
    )


def test_linear_without_comments_create_names_the_scope_and_requests_the_union():
    have = ["read", "write", "issues:create"]
    fields = {"credentials": requirement("linear", {"comments:create"})}
    missing = {
        "credentials": {"provider": "linear", "scopes": ["comments:create"]},
    }

    annotated, messages = annotate_credential_gaps(
        [oauth("linear", have)], fields, missing
    )

    entry = annotated["credentials"]
    # The reconnect asks for what is needed AND what the grant already has:
    # asking for comments:create alone would come back narrower than before.
    assert entry["scopes"] == ["comments:create", "issues:create", "read", "write"]
    assert entry["credential_gap"]["missing_scopes"] == ["comments:create"]
    assert "comments:create" in messages[0]
    assert "is connected" in messages[0]


def test_no_gap_when_a_healthy_credential_fits_or_none_exists():
    need = requirement("github", {"repo"})
    assert find_credential_gap([oauth("github", ["repo"])], need) is None
    assert find_credential_gap([], need) is None
    assert find_credential_gap([oauth("linear", ["read"])], need) is None


# -- A credential whose refresh was refused for good -- #


def test_a_dead_credential_is_a_reconnect_gap_with_the_reason():
    gap = find_credential_gap(
        [oauth("linear", ["read"], metadata=DEAD)], requirement("linear", {"read"})
    )

    assert gap is not None
    assert gap.kind == "reconnect_required"
    assert credential_gap_message("Linear", gap) == (
        "Your saved Linear account 'alice's linear' has to be reconnected: Linear "
        "refused to refresh the saved sign-in (invalid_grant, HTTP 400). "
        "Reconnect it to continue."
    )


def test_matching_skips_a_dead_credential_while_a_healthy_one_fits():
    dead = oauth("linear", ["read"], cred_id="dead", metadata=DEAD)
    live = oauth("linear", ["read"], cred_id="live")
    need = requirement("linear", {"read"})

    assert find_matching_credential([dead, live], need) is live
    # A picked credential that has since died does not win over a live one,
    # and two accounts where one is dead is not a choice to ask about.
    assert find_matching_credential([dead, live], need, {"linear": "dead"}) is live
    assert find_matching_credential([dead, live], need, ask_when_ambiguous=True) is live
    # With nothing else, keep it: the run reaches the reconnect card.
    assert find_matching_credential([dead], need) is dead


def test_rejected_card_says_reconnect_and_why():
    dead = oauth("linear", ["comments:create", "read"], metadata=DEAD)
    marker = reconnect_required(dead)
    assert marker is not None

    card = _build_credential_rejected_card(
        block=LinearCreateCommentBlock(),
        block_id=LinearCreateCommentBlock().id,
        input_data={},
        matched_credentials={
            "credentials": CredentialsMetaInput(
                id=dead.id,
                provider=ProviderName("linear"),
                type="oauth2",
                title=dead.title,
            )
        },
        session_id="session-1",
        status_code=None,
        exc=CredentialsNeedReconnectError("linear", dead.id, marker),
    )

    assert card.message == (
        "The saved Linear credential 'alice's linear' has to be reconnected: "
        "Linear refused to refresh the saved sign-in (invalid_grant, HTTP 400). "
        "Reconnect it or pick a different one, then re-run."
    )
    assert card.rejection is not None
    assert "invalid_grant" in card.rejection.detail
    assert card.setup_info.user_readiness.ready_to_run is False


# -- The connect card the bot relays -- #


async def connect_card(
    existing: list[OAuth2Credentials], provider: str, scopes: list[str]
) -> SetupRequirementsResponse:
    # OAuth is offered only where the GitHub OAuth app is configured, as in prod.
    with patch(
        "backend.copilot.tools.connect_integration.credentials_for_gaps",
        AsyncMock(return_value=existing),
    ), patch(
        "backend.copilot.tools.connect_integration.get_provider_auth_types",
        return_value=["api_key", "oauth2"],
    ):
        result = await ConnectIntegrationTool()._execute(
            user_id=USER,
            session=make_session(user_id=USER),
            provider=provider,
            scopes=scopes,
        )
    assert isinstance(result, SetupRequirementsResponse)
    return result


@pytest.mark.asyncio(loop_scope="session")
async def test_connect_card_names_the_missing_scope_and_requests_the_union():
    card = await connect_card([oauth("github", ["repo"])], "github", ["read:org"])

    assert card.message.startswith(
        "Your GitHub account 'alice's github' is connected, but it was not granted "
        "the read:org permission"
    )
    entry = card.setup_info.user_readiness.missing_credentials["github_credentials"]
    assert entry["scopes"] == ["read:org", "repo"]
    assert entry["credential_gap"]["kind"] == "missing_scopes"
    assert card.setup_info.requirements["credentials"] == [entry]
    assert card.setup_info.user_readiness.has_all_credentials is False


@pytest.mark.asyncio(loop_scope="session")
async def test_connect_card_for_a_never_connected_account_is_unchanged():
    card = await connect_card([], "github", ["read:org"])

    assert card.message.startswith("To continue, please connect your GitHub account.")
    entry = card.setup_info.user_readiness.missing_credentials["github_credentials"]
    assert "credential_gap" not in entry


@pytest.mark.asyncio(loop_scope="session")
async def test_connect_card_survives_an_unreadable_credential_store():
    with patch(
        "backend.copilot.tools.credential_gaps.get_user_credentials",
        AsyncMock(side_effect=RuntimeError("db down")),
    ):
        result = await ConnectIntegrationTool()._execute(
            user_id=USER, session=make_session(user_id=USER), provider="github"
        )

    assert isinstance(result, SetupRequirementsResponse)
    assert result.message.startswith("To continue, please connect")
