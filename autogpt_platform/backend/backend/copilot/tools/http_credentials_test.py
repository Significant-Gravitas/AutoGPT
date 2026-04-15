"""Tests for credential resolution across all credential types in the CoPilot.

These tests verify that:
1. `_resolve_discriminated_credentials` correctly populates discriminator_values
   for URL-based (host-scoped) and provider-based (api_key) credential fields.
2. `find_matching_credential` correctly matches credentials for all types:
   APIKeyCredentials, OAuth2Credentials, UserPasswordCredentials, and
   HostScopedCredentials.
3. The full `resolve_block_credentials` flow correctly resolves matching
   credentials or reports them as missing for each credential type.
4. `RunBlockTool._execute` end-to-end tests return correct response types.
"""

from unittest.mock import AsyncMock, patch

from pydantic import SecretStr

from backend.blocks.http import SendAuthenticatedWebRequestBlock
from backend.data.model import (
    APIKeyCredentials,
    CredentialsFieldInfo,
    CredentialsType,
    HostScopedCredentials,
    OAuth2Credentials,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName

from ._test_data import make_session
from .helpers import _resolve_discriminated_credentials, resolve_block_credentials
from .models import BlockDetailsResponse, SetupRequirementsResponse
from .run_block import RunBlockTool
from .utils import find_matching_credential

_TEST_USER_ID = "test-user-http-cred"

# Properly typed constants to avoid type: ignore on CredentialsFieldInfo construction.
_HOST_SCOPED_TYPES: frozenset[CredentialsType] = frozenset(["host_scoped"])
_API_KEY_TYPES: frozenset[CredentialsType] = frozenset(["api_key"])
_OAUTH2_TYPES: frozenset[CredentialsType] = frozenset(["oauth2"])
_USER_PASSWORD_TYPES: frozenset[CredentialsType] = frozenset(["user_password"])

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
# find_matching_credential tests (api_key)
# ---------------------------------------------------------------------------


class TestFindMatchingAPIKeyCredential:
    """Tests for find_matching_credential with API key credentials."""

    def _make_api_key_cred(
        self, provider: str = "google_maps", cred_id: str = "test-api-key-id"
    ) -> APIKeyCredentials:
        return APIKeyCredentials(
            id=cred_id,
            provider=provider,
            api_key=SecretStr("sk-test-key-123"),
            title=f"API key for {provider}",
            expires_at=None,
        )

    def _make_field_info(
        self, provider: ProviderName = ProviderName.GOOGLE_MAPS
    ) -> CredentialsFieldInfo:
        return CredentialsFieldInfo(
            credentials_provider=frozenset([provider]),
            credentials_types=_API_KEY_TYPES,
            credentials_scopes=None,
        )

    def test_matches_credential_for_correct_provider(self):
        """An API key credential matching the provider should be returned."""
        cred = self._make_api_key_cred("google_maps")
        field_info = self._make_field_info(ProviderName.GOOGLE_MAPS)

        result = find_matching_credential([cred], field_info)
        assert result is not None
        assert result.id == cred.id

    def test_rejects_credential_for_wrong_provider(self):
        """An API key credential for a different provider should not match."""
        cred = self._make_api_key_cred("openai")
        field_info = self._make_field_info(ProviderName.GOOGLE_MAPS)

        result = find_matching_credential([cred], field_info)
        assert result is None

    def test_rejects_credential_for_wrong_type(self):
        """An OAuth2 credential should not match an api_key requirement."""
        oauth_cred = OAuth2Credentials(
            id="oauth-cred-id",
            provider="google_maps",
            access_token=SecretStr("mock-token"),
            scopes=[],
            title="OAuth cred (wrong type)",
        )
        field_info = self._make_field_info(ProviderName.GOOGLE_MAPS)

        result = find_matching_credential([oauth_cred], field_info)
        assert result is None

    def test_selects_correct_credential_from_multiple(self):
        """When multiple API key credentials exist, the correct provider is selected."""
        cred_maps = self._make_api_key_cred("google_maps", "maps-key")
        cred_openai = self._make_api_key_cred("openai", "openai-key")
        field_info = self._make_field_info(ProviderName.OPENAI)

        result = find_matching_credential([cred_maps, cred_openai], field_info)
        assert result is not None
        assert result.id == "openai-key"

    def test_returns_none_when_no_credentials(self):
        """Should return None when the credential list is empty."""
        field_info = self._make_field_info(ProviderName.GOOGLE_MAPS)

        result = find_matching_credential([], field_info)
        assert result is None


# ---------------------------------------------------------------------------
# find_matching_credential tests (oauth2)
# ---------------------------------------------------------------------------


class TestFindMatchingOAuth2Credential:
    """Tests for find_matching_credential with OAuth2 credentials."""

    def _make_oauth2_cred(
        self,
        provider: str = "google",
        scopes: list[str] | None = None,
        cred_id: str = "test-oauth2-id",
    ) -> OAuth2Credentials:
        return OAuth2Credentials(
            id=cred_id,
            provider=provider,
            access_token=SecretStr("mock-access-token"),
            refresh_token=SecretStr("mock-refresh-token"),
            access_token_expires_at=1234567890,
            scopes=scopes or [],
            title=f"OAuth2 cred for {provider}",
        )

    def _make_field_info(
        self,
        provider: ProviderName = ProviderName.GOOGLE,
        required_scopes: frozenset[str] | None = None,
    ) -> CredentialsFieldInfo:
        return CredentialsFieldInfo(
            credentials_provider=frozenset([provider]),
            credentials_types=_OAUTH2_TYPES,
            credentials_scopes=required_scopes,
        )

    def test_matches_credential_for_correct_provider(self):
        """An OAuth2 credential matching the provider should be returned."""
        cred = self._make_oauth2_cred("google")
        field_info = self._make_field_info(ProviderName.GOOGLE)

        result = find_matching_credential([cred], field_info)
        assert result is not None
        assert result.id == cred.id

    def test_rejects_credential_for_wrong_provider(self):
        """An OAuth2 credential for a different provider should not match."""
        cred = self._make_oauth2_cred("github")
        field_info = self._make_field_info(ProviderName.GOOGLE)

        result = find_matching_credential([cred], field_info)
        assert result is None

    def test_matches_credential_with_required_scopes(self):
        """An OAuth2 credential with all required scopes should match."""
        cred = self._make_oauth2_cred(
            "google",
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
            ],
        )
        field_info = self._make_field_info(
            ProviderName.GOOGLE,
            required_scopes=frozenset(
                ["https://www.googleapis.com/auth/gmail.readonly"]
            ),
        )

        result = find_matching_credential([cred], field_info)
        assert result is not None

    def test_rejects_credential_with_insufficient_scopes(self):
        """An OAuth2 credential missing required scopes should not match."""
        cred = self._make_oauth2_cred(
            "google",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        field_info = self._make_field_info(
            ProviderName.GOOGLE,
            required_scopes=frozenset(
                [
                    "https://www.googleapis.com/auth/gmail.readonly",
                    "https://www.googleapis.com/auth/gmail.send",
                ]
            ),
        )

        result = find_matching_credential([cred], field_info)
        assert result is None

    def test_matches_credential_when_no_scopes_required(self):
        """An OAuth2 credential should match when no scopes are required."""
        cred = self._make_oauth2_cred("google", scopes=[])
        field_info = self._make_field_info(ProviderName.GOOGLE)

        result = find_matching_credential([cred], field_info)
        assert result is not None

    def test_selects_correct_credential_from_multiple(self):
        """When multiple OAuth2 credentials exist, the correct one is selected."""
        cred_google = self._make_oauth2_cred("google", cred_id="google-cred")
        cred_github = self._make_oauth2_cred("github", cred_id="github-cred")
        field_info = self._make_field_info(ProviderName.GITHUB)

        result = find_matching_credential([cred_google, cred_github], field_info)
        assert result is not None
        assert result.id == "github-cred"

    def test_returns_none_when_no_credentials(self):
        """Should return None when the credential list is empty."""
        field_info = self._make_field_info(ProviderName.GOOGLE)

        result = find_matching_credential([], field_info)
        assert result is None


# ---------------------------------------------------------------------------
# find_matching_credential tests (user_password)
# ---------------------------------------------------------------------------


class TestFindMatchingUserPasswordCredential:
    """Tests for find_matching_credential with user/password credentials."""

    def _make_user_password_cred(
        self, provider: str = "smtp", cred_id: str = "test-userpass-id"
    ) -> UserPasswordCredentials:
        return UserPasswordCredentials(
            id=cred_id,
            provider=provider,
            username=SecretStr("test-user"),
            password=SecretStr("test-pass"),
            title=f"Credentials for {provider}",
        )

    def _make_field_info(
        self, provider: ProviderName = ProviderName.SMTP
    ) -> CredentialsFieldInfo:
        return CredentialsFieldInfo(
            credentials_provider=frozenset([provider]),
            credentials_types=_USER_PASSWORD_TYPES,
            credentials_scopes=None,
        )

    def test_matches_credential_for_correct_provider(self):
        """A user/password credential matching the provider should be returned."""
        cred = self._make_user_password_cred("smtp")
        field_info = self._make_field_info(ProviderName.SMTP)

        result = find_matching_credential([cred], field_info)
        assert result is not None
        assert result.id == cred.id

    def test_rejects_credential_for_wrong_provider(self):
        """A user/password credential for a different provider should not match."""
        cred = self._make_user_password_cred("smtp")
        field_info = self._make_field_info(ProviderName.HUBSPOT)

        result = find_matching_credential([cred], field_info)
        assert result is None

    def test_rejects_credential_for_wrong_type(self):
        """An API key credential should not match a user_password requirement."""
        api_key_cred = APIKeyCredentials(
            id="api-key-cred-id",
            provider="smtp",
            api_key=SecretStr("wrong-type-key"),
            title="API key cred (wrong type)",
        )
        field_info = self._make_field_info(ProviderName.SMTP)

        result = find_matching_credential([api_key_cred], field_info)
        assert result is None

    def test_selects_correct_credential_from_multiple(self):
        """When multiple user/password credentials exist, the correct one is selected."""
        cred_smtp = self._make_user_password_cred("smtp", "smtp-cred")
        cred_hubspot = self._make_user_password_cred("hubspot", "hubspot-cred")
        field_info = self._make_field_info(ProviderName.HUBSPOT)

        result = find_matching_credential([cred_smtp, cred_hubspot], field_info)
        assert result is not None
        assert result.id == "hubspot-cred"

    def test_returns_none_when_no_credentials(self):
        """Should return None when the credential list is empty."""
        field_info = self._make_field_info(ProviderName.SMTP)

        result = find_matching_credential([], field_info)
        assert result is None


# ---------------------------------------------------------------------------
# find_matching_credential tests (mixed credential types)
# ---------------------------------------------------------------------------


class TestFindMatchingCredentialMixedTypes:
    """Tests that find_matching_credential correctly filters by type in a mixed list."""

    def test_selects_api_key_from_mixed_list(self):
        """API key requirement should skip OAuth2 and user_password credentials."""
        oauth_cred = OAuth2Credentials(
            id="oauth-id",
            provider="openai",
            access_token=SecretStr("token"),
            scopes=[],
        )
        userpass_cred = UserPasswordCredentials(
            id="userpass-id",
            provider="openai",
            username=SecretStr("user"),
            password=SecretStr("pass"),
        )
        api_key_cred = APIKeyCredentials(
            id="apikey-id",
            provider="openai",
            api_key=SecretStr("sk-key"),
        )
        field_info = CredentialsFieldInfo(
            credentials_provider=frozenset([ProviderName.OPENAI]),
            credentials_types=_API_KEY_TYPES,
            credentials_scopes=None,
        )

        result = find_matching_credential(
            [oauth_cred, userpass_cred, api_key_cred], field_info
        )
        assert result is not None
        assert result.id == "apikey-id"

    def test_selects_oauth2_from_mixed_list(self):
        """OAuth2 requirement should skip API key and user_password credentials."""
        api_key_cred = APIKeyCredentials(
            id="apikey-id",
            provider="google",
            api_key=SecretStr("key"),
        )
        userpass_cred = UserPasswordCredentials(
            id="userpass-id",
            provider="google",
            username=SecretStr("user"),
            password=SecretStr("pass"),
        )
        oauth_cred = OAuth2Credentials(
            id="oauth-id",
            provider="google",
            access_token=SecretStr("token"),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        field_info = CredentialsFieldInfo(
            credentials_provider=frozenset([ProviderName.GOOGLE]),
            credentials_types=_OAUTH2_TYPES,
            credentials_scopes=frozenset(
                ["https://www.googleapis.com/auth/gmail.readonly"]
            ),
        )

        result = find_matching_credential(
            [api_key_cred, userpass_cred, oauth_cred], field_info
        )
        assert result is not None
        assert result.id == "oauth-id"

    def test_selects_user_password_from_mixed_list(self):
        """User/password requirement should skip API key and OAuth2 credentials."""
        api_key_cred = APIKeyCredentials(
            id="apikey-id",
            provider="smtp",
            api_key=SecretStr("key"),
        )
        oauth_cred = OAuth2Credentials(
            id="oauth-id",
            provider="smtp",
            access_token=SecretStr("token"),
            scopes=[],
        )
        userpass_cred = UserPasswordCredentials(
            id="userpass-id",
            provider="smtp",
            username=SecretStr("user"),
            password=SecretStr("pass"),
        )
        field_info = CredentialsFieldInfo(
            credentials_provider=frozenset([ProviderName.SMTP]),
            credentials_types=_USER_PASSWORD_TYPES,
            credentials_scopes=None,
        )

        result = find_matching_credential(
            [api_key_cred, oauth_cred, userpass_cred], field_info
        )
        assert result is not None
        assert result.id == "userpass-id"

    def test_returns_none_when_only_wrong_types_available(self):
        """Should return None when all available creds have the wrong type."""
        oauth_cred = OAuth2Credentials(
            id="oauth-id",
            provider="google_maps",
            access_token=SecretStr("token"),
            scopes=[],
        )
        field_info = CredentialsFieldInfo(
            credentials_provider=frozenset([ProviderName.GOOGLE_MAPS]),
            credentials_types=_API_KEY_TYPES,
            credentials_scopes=None,
        )

        result = find_matching_credential([oauth_cred], field_info)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_block_credentials tests (integration — all credential types)
# ---------------------------------------------------------------------------


class TestResolveBlockCredentials:
    """Integration tests for resolve_block_credentials across credential types."""

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

    async def test_matches_api_key_credential_for_llm_block(self):
        """resolve_block_credentials should match an API key cred for an LLM block."""
        from backend.blocks.llm import AITextGeneratorBlock

        block = AITextGeneratorBlock()
        input_data = {"model": "gpt-4o-mini"}

        mock_cred = APIKeyCredentials(
            id="openai-key-id",
            provider="openai",
            api_key=SecretStr("sk-test-key"),
            title="OpenAI API Key",
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
        assert matched["credentials"].id == "openai-key-id"
        assert len(missing) == 0

    async def test_reports_missing_api_key_for_wrong_provider(self):
        """resolve_block_credentials should report missing when API key provider mismatches."""
        from backend.blocks.llm import AITextGeneratorBlock

        block = AITextGeneratorBlock()
        input_data = {"model": "gpt-4o-mini"}

        wrong_provider_cred = APIKeyCredentials(
            id="wrong-key-id",
            provider="google_maps",
            api_key=SecretStr("sk-wrong"),
            title="Google Maps Key",
        )

        with patch(
            "backend.copilot.tools.utils.get_user_credentials",
            new_callable=AsyncMock,
            return_value=[wrong_provider_cred],
        ):
            matched, missing = await resolve_block_credentials(
                _TEST_USER_ID, block, input_data
            )

        assert len(matched) == 0
        assert len(missing) == 1

    async def test_matches_oauth2_credential_for_google_block(self):
        """resolve_block_credentials should match an OAuth2 cred for a Google block."""
        from backend.blocks.google.gmail import GmailReadBlock

        block = GmailReadBlock()
        input_data = {}

        mock_cred = OAuth2Credentials(
            id="google-oauth-id",
            provider="google",
            access_token=SecretStr("mock-token"),
            refresh_token=SecretStr("mock-refresh"),
            access_token_expires_at=9999999999,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            title="Google OAuth",
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
        assert matched["credentials"].id == "google-oauth-id"
        assert len(missing) == 0

    async def test_reports_missing_oauth2_with_insufficient_scopes(self):
        """resolve_block_credentials should report missing when OAuth2 scopes are insufficient."""
        from backend.blocks.google.gmail import GmailSendBlock

        block = GmailSendBlock()
        input_data = {}

        # GmailSendBlock requires gmail.send scope; provide only readonly
        insufficient_cred = OAuth2Credentials(
            id="limited-oauth-id",
            provider="google",
            access_token=SecretStr("mock-token"),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            title="Google OAuth (limited)",
        )

        with patch(
            "backend.copilot.tools.utils.get_user_credentials",
            new_callable=AsyncMock,
            return_value=[insufficient_cred],
        ):
            matched, missing = await resolve_block_credentials(
                _TEST_USER_ID, block, input_data
            )

        assert len(matched) == 0
        assert len(missing) == 1

    async def test_matches_user_password_credential_for_email_block(self):
        """resolve_block_credentials should match a user/password cred for an SMTP block."""
        from backend.blocks.email_block import SendEmailBlock

        block = SendEmailBlock()
        input_data = {}

        mock_cred = UserPasswordCredentials(
            id="smtp-cred-id",
            provider="smtp",
            username=SecretStr("test-user"),
            password=SecretStr("test-pass"),
            title="SMTP Credentials",
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
        assert matched["credentials"].id == "smtp-cred-id"
        assert len(missing) == 0

    async def test_reports_missing_user_password_for_wrong_provider(self):
        """resolve_block_credentials should report missing when user/password provider mismatches."""
        from backend.blocks.email_block import SendEmailBlock

        block = SendEmailBlock()
        input_data = {}

        wrong_cred = UserPasswordCredentials(
            id="wrong-cred-id",
            provider="dataforseo",
            username=SecretStr("user"),
            password=SecretStr("pass"),
            title="DataForSEO Creds",
        )

        with patch(
            "backend.copilot.tools.utils.get_user_credentials",
            new_callable=AsyncMock,
            return_value=[wrong_cred],
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
