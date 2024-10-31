import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from pydantic import SecretStr

if TYPE_CHECKING:
    from redis import Redis
    from backend.executor.database import DatabaseManager

from autogpt_libs.utils.cache import thread_cached
from autogpt_libs.utils.synchronize import RedisKeyedMutex

from .types import (
    APIKeyCredentials,
    Credentials,
    OAuth2Credentials,
    OAuthState,
    UserIntegrations,
)

from backend.util.settings import Settings

settings = Settings()

revid_credentials = APIKeyCredentials(
    id="fdb7f412-f519-48d1-9b5f-d2f73d0e01fe",
    provider="revid",
    api_key=SecretStr(settings.secrets.revid_api_key),
    title="Use Credits for Revid",
    expires_at=None,
)
ideogram_credentials = APIKeyCredentials(
    id="760f84fc-b270-42de-91f6-08efe1b512d0",
    provider="ideogram",
    api_key=SecretStr(settings.secrets.ideogram_api_key),
    title="Use Credits for Ideogram",
    expires_at=None,
)
replicate_credentials = APIKeyCredentials(
    id="6b9fc200-4726-4973-86c9-cd526f5ce5db",
    provider="replicate",
    api_key=SecretStr(settings.secrets.replicate_api_key),
    title="Use Credits for Replicate",
    expires_at=None,
)
openai_credentials = APIKeyCredentials(
    id="53c25cb8-e3ee-465c-a4d1-e75a4c899c2a",
    provider="llm",
    api_key=SecretStr(settings.secrets.openai_api_key),
    title="Use Credits for OpenAI",
    expires_at=None,
)
anthropic_credentials = APIKeyCredentials(
    id="24e5d942-d9e3-4798-8151-90143ee55629",
    provider="llm",
    api_key=SecretStr(settings.secrets.anthropic_api_key),
    title="Use Credits for Anthropic",
    expires_at=None,
)
groq_credentials = APIKeyCredentials(
    id="4ec22295-8f97-4dd1-b42b-2c6957a02545",
    provider="llm",
    api_key=SecretStr(settings.secrets.groq_api_key),
    title="Use Credits for Groq",
    expires_at=None,
)
did_credentials = APIKeyCredentials(
    id="7f7b0654-c36b-4565-8fa7-9a52575dfae2",
    provider="d_id",
    api_key=SecretStr(settings.secrets.did_api_key),
    title="Use Credits for D-ID",
    expires_at=None,
)

DEFAULT_CREDENTIALS = [
    revid_credentials,
    ideogram_credentials,
    replicate_credentials,
    openai_credentials,
    anthropic_credentials,
    groq_credentials,
    did_credentials,
]


class SupabaseIntegrationCredentialsStore:
    def __init__(self, redis: "Redis"):
        self.locks = RedisKeyedMutex(redis)

    @property
    @thread_cached
    def db_manager(self) -> "DatabaseManager":
        from backend.executor.database import DatabaseManager
        from backend.util.service import get_service_client

        return get_service_client(DatabaseManager)

    def add_creds(self, user_id: str, credentials: Credentials) -> None:
        with self.locked_user_integrations(user_id):
            if self.get_creds_by_id(user_id, credentials.id):
                raise ValueError(
                    f"Can not re-create existing credentials #{credentials.id} "
                    f"for user #{user_id}"
                )
            self._set_user_integration_creds(
                user_id, [*self.get_all_creds(user_id), credentials]
            )

    def get_all_creds(self, user_id: str) -> list[Credentials]:
        users_credentials = self._get_user_integrations(user_id).credentials
        all_credentials = users_credentials
        if settings.secrets.revid_api_key:
            all_credentials.append(revid_credentials)
        if settings.secrets.ideogram_api_key:
            all_credentials.append(ideogram_credentials)
        if settings.secrets.groq_api_key:
            all_credentials.append(groq_credentials)
        if settings.secrets.replicate_api_key:
            all_credentials.append(replicate_credentials)
        if settings.secrets.openai_api_key:
            all_credentials.append(openai_credentials)
        if settings.secrets.anthropic_api_key:
            all_credentials.append(anthropic_credentials)
        if settings.secrets.did_api_key:
            all_credentials.append(did_credentials)
        return all_credentials

    def get_creds_by_id(self, user_id: str, credentials_id: str) -> Credentials | None:
        all_credentials = self.get_all_creds(user_id)
        return next((c for c in all_credentials if c.id == credentials_id), None)

    def get_creds_by_provider(self, user_id: str, provider: str) -> list[Credentials]:
        credentials = self.get_all_creds(user_id)
        return [c for c in credentials if c.provider == provider]

    def get_authorized_providers(self, user_id: str) -> list[str]:
        credentials = self.get_all_creds(user_id)
        return list(set(c.provider for c in credentials))

    def update_creds(self, user_id: str, updated: Credentials) -> None:
        with self.locked_user_integrations(user_id):
            current = self.get_creds_by_id(user_id, updated.id)
            if not current:
                raise ValueError(
                    f"Credentials with ID {updated.id} "
                    f"for user with ID {user_id} not found"
                )
            if type(current) is not type(updated):
                raise TypeError(
                    f"Can not update credentials with ID {updated.id} "
                    f"from type {type(current)} "
                    f"to type {type(updated)}"
                )

            # Ensure no scopes are removed when updating credentials
            if (
                isinstance(updated, OAuth2Credentials)
                and isinstance(current, OAuth2Credentials)
                and not set(updated.scopes).issuperset(current.scopes)
            ):
                raise ValueError(
                    f"Can not update credentials with ID {updated.id} "
                    f"and scopes {current.scopes} "
                    f"to more restrictive set of scopes {updated.scopes}"
                )

            # Update the credentials
            updated_credentials_list = [
                updated if c.id == updated.id else c
                for c in self.get_all_creds(user_id)
            ]
            self._set_user_integration_creds(user_id, updated_credentials_list)

    def delete_creds_by_id(self, user_id: str, credentials_id: str) -> None:
        with self.locked_user_integrations(user_id):
            filtered_credentials = [
                c for c in self.get_all_creds(user_id) if c.id != credentials_id
            ]
            self._set_user_integration_creds(user_id, filtered_credentials)

    def store_state_token(self, user_id: str, provider: str, scopes: list[str]) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        state = OAuthState(
            token=token,
            provider=provider,
            expires_at=int(expires_at.timestamp()),
            scopes=scopes,
        )

        with self.locked_user_integrations(user_id):
            user_integrations = self._get_user_integrations(user_id)
            oauth_states = user_integrations.oauth_states
            oauth_states.append(state)
            user_integrations.oauth_states = oauth_states

            self.db_manager.update_user_integrations(
                user_id=user_id, data=user_integrations
            )

        return token

    def get_any_valid_scopes_from_state_token(
        self, user_id: str, token: str, provider: str
    ) -> list[str]:
        """
        Get the valid scopes from the OAuth state token. This will return any valid scopes
        from any OAuth state token for the given provider. If no valid scopes are found,
        an empty list is returned. DO NOT RELY ON THIS TOKEN TO AUTHENTICATE A USER, AS IT
        IS TO CHECK IF THE USER HAS GIVEN PERMISSIONS TO THE APPLICATION BEFORE EXCHANGING
        THE CODE FOR TOKENS.
        """
        user_integrations = self._get_user_integrations(user_id)
        oauth_states = user_integrations.oauth_states

        now = datetime.now(timezone.utc)
        valid_state = next(
            (
                state
                for state in oauth_states
                if state.token == token
                and state.provider == provider
                and state.expires_at > now.timestamp()
            ),
            None,
        )

        if valid_state:
            return valid_state.scopes

        return []

    def verify_state_token(self, user_id: str, token: str, provider: str) -> bool:
        with self.locked_user_integrations(user_id):
            user_integrations = self._get_user_integrations(user_id)
            oauth_states = user_integrations.oauth_states

            now = datetime.now(timezone.utc)
            valid_state = next(
                (
                    state
                    for state in oauth_states
                    if state.token == token
                    and state.provider == provider
                    and state.expires_at > now.timestamp()
                ),
                None,
            )

            if valid_state:
                # Remove the used state
                oauth_states.remove(valid_state)
                user_integrations.oauth_states = oauth_states
                self.db_manager.update_user_integrations(user_id, user_integrations)
                return True

        return False

    def _set_user_integration_creds(
        self, user_id: str, credentials: list[Credentials]
    ) -> None:
        integrations = self._get_user_integrations(user_id)
        # Remove default credentials from the list
        credentials = [c for c in credentials if c not in DEFAULT_CREDENTIALS]
        integrations.credentials = credentials
        self.db_manager.update_user_integrations(user_id, integrations)

    def _get_user_integrations(self, user_id: str) -> UserIntegrations:
        integrations: UserIntegrations = self.db_manager.get_user_integrations(
            user_id=user_id
        )
        return integrations

    def locked_user_integrations(self, user_id: str):
        key = (self.db_manager, f"user:{user_id}", "integrations")
        return self.locks.locked(key)
