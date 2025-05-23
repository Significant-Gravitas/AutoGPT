import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from pydantic import SecretStr

if TYPE_CHECKING:
    from backend.executor.database import DatabaseManagerClient

import functools
import importlib
import pkgutil
from pathlib import Path

from autogpt_libs.utils.cache import thread_cached
from autogpt_libs.utils.synchronize import RedisKeyedMutex

from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    OAuth2Credentials,
    OAuthState,
    UserIntegrations,
)
from backend.util.settings import Settings

settings = Settings()

# This provider does not require a real API key but the credential system expects one
ollama_credentials = APIKeyCredentials(
    id="744fdc56-071a-4761-b5a5-0af0ce10a2b5",
    provider="ollama",
    api_key=SecretStr("FAKE_API_KEY"),
    title="Use Credits for Ollama",
    expires_at=None,
)


def iter_block_modules(base_package: str):
    """Yield all modules within a block package recursively."""

    pkg = importlib.import_module(base_package)
    assert pkg.__file__
    base_path = Path(pkg.__file__).parent
    for mod_info in pkgutil.walk_packages([str(base_path)], prefix=f"{base_package}."):
        yield importlib.import_module(mod_info.name)


@functools.cache
def discover_default_credentials() -> list[APIKeyCredentials]:
    defaults: list[APIKeyCredentials] = []
    for mod in iter_block_modules("backend.blocks"):
        if hasattr(mod, "default_credentials"):
            cred = mod.default_credentials()
            if cred:
                defaults.append(cred)
    return defaults


DEFAULT_CREDENTIALS = [ollama_credentials, *discover_default_credentials()]


class IntegrationCredentialsStore:
    def __init__(self):
        from backend.data.redis import get_redis

        self.locks = RedisKeyedMutex(get_redis())

    @property
    @thread_cached
    def db_manager(self) -> "DatabaseManagerClient":
        from backend.executor.database import DatabaseManagerClient
        from backend.util.service import get_service_client

        return get_service_client(DatabaseManagerClient)

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
        return [*users_credentials, *DEFAULT_CREDENTIALS]

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

    def store_state_token(
        self, user_id: str, provider: str, scopes: list[str], use_pkce: bool = False
    ) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        (code_challenge, code_verifier) = self._generate_code_challenge()

        state = OAuthState(
            token=token,
            provider=provider,
            code_verifier=code_verifier,
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

        return token, code_challenge

    def _generate_code_challenge(self) -> tuple[str, str]:
        """
        Generate code challenge using SHA256 from the code verifier.
        Currently only SHA256 is supported.(In future if we want to support more methods we can add them here)
        """
        code_verifier = secrets.token_urlsafe(128)
        sha256_hash = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")
        return code_challenge.replace("=", ""), code_verifier

    def verify_state_token(
        self, user_id: str, token: str, provider: str
    ) -> Optional[OAuthState]:
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
                return valid_state

        return None

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
        key = (f"user:{user_id}", "integrations")
        return self.locks.locked(key)
