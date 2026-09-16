"""Characterization tests for narrow-scope credential handling.

Covers the investigation/click-button-correlation finding: a stored
credential that lacks one required scope is reported missing, and the
setup-card payload carries no reference to the existing row, so the UI
can only offer a fresh connect instead of a scope upgrade.

When the upgrade affordance ships, UPDATE these tests to assert the
payload references the under-scoped row.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.linear._config import LinearScope
from backend.blocks.linear.comment import LinearCreateCommentBlock
from backend.copilot.tools import helpers as block_helpers
from backend.copilot.tools import utils as tool_utils
from backend.copilot.tools.models import SetupRequirementsResponse
from backend.data.model import CredentialsFieldInfo, CredentialsType, OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.sdk import SecretStr

from ._test_data import make_session

_NARROW_ROW_ID = "cred-narrow-linear-1"


def _narrow_linear_cred() -> OAuth2Credentials:
    """Linear OAuth row with everything EXCEPT comments:create."""
    return OAuth2Credentials(
        id=_NARROW_ROW_ID,
        provider="linear",
        title="Probe Linear OAuth",
        username="probe-user",
        access_token=SecretStr("mock-access-token"),
        access_token_expires_at=1999999999,
        refresh_token=SecretStr("mock-refresh-token"),
        refresh_token_expires_at=None,
        scopes=["read", "write", "issues:create"],
    )


def _comment_field_info() -> CredentialsFieldInfo:
    """Mirrors LinearCreateCommentBlock's credential declaration."""
    return CredentialsFieldInfo[ProviderName, CredentialsType](
        credentials_provider=frozenset({ProviderName("linear")}),
        credentials_types=frozenset({"oauth2"}),
        credentials_scopes=frozenset({LinearScope.COMMENTS_CREATE.value}),
    )


def test_narrow_scope_row_fails_superset_check():
    field_info = _comment_field_info()
    assert not tool_utils._credential_has_required_scopes(
        _narrow_linear_cred(), field_info
    )


@pytest.mark.asyncio
async def test_narrow_scope_row_reports_missing_without_upgrade_pointer():
    """Full block path: real block, stubbed store, real matcher+serializer."""
    real_block = LinearCreateCommentBlock()
    with (
        patch.object(block_helpers, "get_block", return_value=real_block),
        patch.object(
            tool_utils,
            "get_user_credentials",
            new=AsyncMock(return_value=[_narrow_linear_cred()]),
        ),
    ):
        result = await block_helpers.prepare_block_for_execution(
            block_id=real_block.id,
            input_data={"issue_id": "SENTRY-1394", "comment": "probe"},
            user_id="user-probe",
            session=make_session("user-probe"),
            session_id="sess-probe",
            dry_run=False,
        )
    assert isinstance(result, SetupRequirementsResponse)
    assert "LinearCreateCommentBlock" in result.message
    assert not result.setup_info.user_readiness.has_all_credentials
    payload = result.setup_info.user_readiness.missing_credentials
    assert payload["credentials"]["scopes"] == ["comments:create"]
    # The defect: no reference to the existing under-scoped row, so the
    # card cannot offer a scope upgrade of it.
    assert _NARROW_ROW_ID not in str(payload)
