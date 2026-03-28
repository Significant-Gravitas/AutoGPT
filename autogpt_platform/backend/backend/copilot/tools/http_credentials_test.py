"""Tests for authenticated HTTP web request credential resolution in the CoPilot.

These tests verify that:
1. `_resolve_discriminated_credentials` correctly populates discriminator_values
   for URL-based (host-scoped) credential fields.
2. `find_matching_credential` correctly matches host-scoped credentials by URL.
3. The full `prepare_block_for_execution` flow for `SendAuthenticatedWebRequestBlock`
   correctly resolves matching credentials or reports them as missing.
"""

from unittest.mock import AsyncMock, patch

from pydantic import SecretStr

from backend.blocks.http import SendAuthenticatedWebRequestBlock
from backend.data.model import (
    CredentialsFieldInfo,
    CredentialsType,
    HostScopedCredentials,
)
from backend.integrations.providers import ProviderName

from ._test_data import make_session
from .helpers import _resolve_discriminated_credentials, resolve_block_credentials
from .models import BlockDetailsResponse, SetupRequirementsResponse
from .run_block import RunBlockTool
from .utils import find_matching_credential

_TEST_USER_ID = "test-user-http-cred"

# Properly typed constant to avoid type: ignore on CredentialsFieldInfo construction.
_HOST_SCOPED_TYPES: frozenset[CredentialsType] = frozenset(["host_scoped"])

# ---------------------------------------------------------------------------
# _resolve_discriminated_credentials tests
# ---------------------------------------------------------------------------


class TestResolveDiscriminatedCredentials:
    """Tests for _resolve_discriminated_credentials with URL-based discrimination."""

    def _get_auth_block(self):
        return SendAuthenticatedWebRequestBlock()

    def test_url_discriminator_populates_discriminator_values(self):
        """When input_data contains a URL, discriminator_values should include it."""
        block = self._get_auth_block()
        input_data = {"url": "https://api.example.com/v1/data"}

        result = _resolve_discriminated_credentials(block, input_data)

        assert "credentials" in result
        field_info = result["credentials"]
        assert "https://api.example.com/v1/data" in field_info.discriminator_values

    def test_url_discriminator_without_url_keeps_empty_values(self):
        """When no URL is provided, discriminator_values should remain empty."""
        block = self._get_auth_block()
        input_data = {}

        result = _resolve_discriminated_credentials(block, input_data)

        assert "credentials" in result
        field_info = result["credentials"]
        assert len(field_info.discriminator_values) == 0

    def test_url_discriminator_does_not_mutate_original_field_info(self):
        """The original block schema field_info must not be mutated."""
        block = self._get_auth_block()

        # Grab a reference to the original schema-level field_info
        original_info = block.input_schema.get_credentials_fields_info()["credentials"]

        # Call with a URL, which adds to discriminator_values on the copy
        _resolve_discriminated_credentials(
            block, {"url": "https://api.example.com/v1/data"}
        )

        # The original object must remain unchanged
        assert len(original_info.discriminator_values) == 0

        # And a fresh call without URL should also return empty values
        result = _resolve_discriminated_credentials(block, {})
        field_info = result["credentials"]
        assert len(field_info.discriminator_values) == 0

    def test_url_discriminator_preserves_provider_and_type(self):
        """Provider and supported_types should be preserved after URL discrimination."""
        block = self._get_auth_block()
        input_data = {"url": "https://api.example.com/v1/data"}

        result = _resolve_discriminated_credentials(block, input_data)

        field_info = result["credentials"]
        assert ProviderName.HTTP in field_info.provider
        assert "host_scoped" in field_info.supported_types

    def test_provider_discriminator_still_works(self):
        """Verify provider-based discrimination (e.g. model -> provider) is preserved.

        The refactored conditional in _resolve_discriminated_credentials split the
        original single ``if`` into nested ``if/else`` branches. This test ensures
        the provider-based path still narrows the provider correctly.
        """
        from backend.blocks.llm import AITextGeneratorBlock

        block = AITextGeneratorBlock()
        input_data = {"model": "gpt-4o-mini"}

        result = _resolve_discriminated_credentials(block, input_data)

        assert "credentials" in result
        field_info = result["credentials"]
        # Should narrow provider to openai
        assert ProviderName.OPENAI in field_info.provider
        assert "gpt-4o-mini" in field_info.discriminator_values


# ---------------------------------------------------------------------------
# find_matching_credential tests (host-scoped)
# ---------------------------------------------------------------------------


class TestFindMatchingHostScopedCredential:
    """Tests for find_matching_credential with host-scoped credentials."""

    def _make_host_scoped_cred(
        self, host: str, cred_id: str = "test-cred-id"
    ) -> HostScopedCredentials:
        return HostScopedCredentials(
            id=cred_id,
            provider="http",
            host=host,
            headers={"Authorization": SecretStr("Bearer test-token")},
            title=f"Cred for {host}",
        )

    def _make_field_info(
        self, discriminator_values: set | None = None
    ) -> CredentialsFieldInfo:
        return CredentialsFieldInfo(
            credentials_provider=frozenset([ProviderName.HTTP]),
            credentials_types=_HOST_SCOPED_TYPES,
            credentials_scopes=None,
            discriminator="url",
            discriminator_values=discriminator_values or set(),
        )

    def test_matches_credential_for_correct_host(self):
        """A host-scoped credential matching the URL host should be returned."""
        cred = self._make_host_scoped_cred("api.example.com")
        field_info = self._make_field_info({"https://api.example.com/v1/data"})

        result = find_matching_credential([cred], field_info)
        assert result is not None
        assert result.id == cred.id

    def test_rejects_credential_for_wrong_host(self):
        """A host-scoped credential for a different host should not match."""
        cred = self._make_host_scoped_cred("api.github.com")
        field_info = self._make_field_info({"https://api.stripe.com/v1/charges"})

        result = find_matching_credential([cred], field_info)
        assert result is None

    def test_matches_any_when_no_discriminator_values(self):
        """With empty discriminator_values, any host-scoped credential matches.

        Note: this tests the current fallback behavior in _credential_is_for_host()
        where empty discriminator_values means "no host constraint" and any
        host-scoped credential is accepted. This is by design for the case where
        the target URL is not yet known (e.g. schema preview with empty input).
        """
        cred = self._make_host_scoped_cred("api.anything.com")
        field_info = self._make_field_info(set())

        result = find_matching_credential([cred], field_info)
        assert result is not None

    def test_wildcard_host_matching(self):
        """Wildcard host (*.example.com) should match subdomains."""
        cred = self._make_host_scoped_cred("*.example.com")
        field_info = self._make_field_info({"https://api.example.com/v1/data"})

        result = find_matching_credential([cred], field_info)
        assert result is not None

    def test_selects_correct_credential_from_multiple(self):
        """When multiple host-scoped credentials exist, the correct one is selected."""
        cred_github = self._make_host_scoped_cred("api.github.com", "github-cred")
        cred_stripe = self._make_host_scoped_cred("api.stripe.com", "stripe-cred")
        field_info = self._make_field_info({"https://api.stripe.com/v1/charges"})

        result = find_matching_credential([cred_github, cred_stripe], field_info)
        assert result is not None
        assert result.id == "stripe-cred"


# ---------------------------------------------------------------------------
# resolve_block_credentials tests (integration)
# ---------------------------------------------------------------------------


class TestResolveBlockCredentials:
    """Integration tests for resolve_block_credentials with HTTP blocks."""

    async def test_matches_host_scoped_credential_for_url(self):
        """resolve_block_credentials should match a host-scoped cred for the given URL."""
        block = SendAuthenticatedWebRequestBlock()
        input_data = {"url": "https://api.example.com/v1/data"}

        mock_cred = HostScopedCredentials(
            id="matching-cred-id",
            provider="http",
            host="api.example.com",
            headers={"Authorization": SecretStr("Bearer token")},
            title="Example API Cred",
        )

        with patch(
            "backend.copilot.tools.utils.get_user_credentials",
            new_callable=AsyncMock,
            return_value=[mock_cred],
        ):
            matched, missing = await resolve_block_credentials(
                _TEST_USER_ID, block, input_data
            )

        assert "credentials" in matched
        assert matched["credentials"].id == "matching-cred-id"
        assert len(missing) == 0

    async def test_reports_missing_when_no_matching_host(self):
        """resolve_block_credentials should report missing creds when host doesn't match."""
        block = SendAuthenticatedWebRequestBlock()
        input_data = {"url": "https://api.stripe.com/v1/charges"}

        wrong_host_cred = HostScopedCredentials(
            id="wrong-cred-id",
            provider="http",
            host="api.github.com",
            headers={"Authorization": SecretStr("Bearer token")},
            title="GitHub API Cred",
        )

        with patch(
            "backend.copilot.tools.utils.get_user_credentials",
            new_callable=AsyncMock,
            return_value=[wrong_host_cred],
        ):
            matched, missing = await resolve_block_credentials(
                _TEST_USER_ID, block, input_data
            )

        assert len(matched) == 0
        assert len(missing) == 1

    async def test_reports_missing_when_no_credentials(self):
        """resolve_block_credentials should report missing when user has no creds at all."""
        block = SendAuthenticatedWebRequestBlock()
        input_data = {"url": "https://api.example.com/v1/data"}

        with patch(
            "backend.copilot.tools.utils.get_user_credentials",
            new_callable=AsyncMock,
            return_value=[],
        ):
            matched, missing = await resolve_block_credentials(
                _TEST_USER_ID, block, input_data
            )

        assert len(matched) == 0
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# RunBlockTool integration tests for authenticated HTTP
# ---------------------------------------------------------------------------


class TestRunBlockToolAuthenticatedHttp:
    """End-to-end tests for RunBlockTool with SendAuthenticatedWebRequestBlock."""

    async def test_returns_setup_requirements_when_creds_missing(self):
        """When no matching host-scoped credential exists, return SetupRequirementsResponse."""
        session = make_session(user_id=_TEST_USER_ID)
        block = SendAuthenticatedWebRequestBlock()

        with patch(
            "backend.copilot.tools.helpers.get_block",
            return_value=block,
        ):
            with patch(
                "backend.copilot.tools.utils.get_user_credentials",
                new_callable=AsyncMock,
                return_value=[],
            ):
                tool = RunBlockTool()
                response = await tool._execute(
                    user_id=_TEST_USER_ID,
                    session=session,
                    block_id=block.id,
                    input_data={"url": "https://api.example.com/data", "method": "GET"},
                    dry_run=False,
                )

        assert isinstance(response, SetupRequirementsResponse)
        assert "credentials" in response.message.lower()

    async def test_returns_details_when_creds_matched_but_missing_required_inputs(self):
        """When creds present + required inputs missing -> BlockDetailsResponse.

        Note: with input_data={}, no URL is provided so discriminator_values is
        empty, meaning _credential_is_for_host() matches any host-scoped
        credential vacuously. This test exercises the "creds present + inputs
        missing" branch, not host-based matching (which is covered by
        TestFindMatchingHostScopedCredential and TestResolveBlockCredentials).
        """
        session = make_session(user_id=_TEST_USER_ID)
        block = SendAuthenticatedWebRequestBlock()

        mock_cred = HostScopedCredentials(
            id="matching-cred-id",
            provider="http",
            host="api.example.com",
            headers={"Authorization": SecretStr("Bearer token")},
            title="Example API Cred",
        )

        with patch(
            "backend.copilot.tools.helpers.get_block",
            return_value=block,
        ):
            with patch(
                "backend.copilot.tools.utils.get_user_credentials",
                new_callable=AsyncMock,
                return_value=[mock_cred],
            ):
                tool = RunBlockTool()
                # Call with empty input to get schema
                response = await tool._execute(
                    user_id=_TEST_USER_ID,
                    session=session,
                    block_id=block.id,
                    input_data={},
                    dry_run=False,
                )

        assert isinstance(response, BlockDetailsResponse)
        assert response.block.name == block.name
        # The matched credential should be included in the details
        assert len(response.block.credentials) > 0
        assert response.block.credentials[0].id == "matching-cred-id"
