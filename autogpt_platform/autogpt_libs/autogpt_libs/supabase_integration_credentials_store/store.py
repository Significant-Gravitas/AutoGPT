import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from redis import Redis
    from supabase import Client

from autogpt_libs.utils.synchronize import RedisKeyedMutex

from .types import (
    Credentials,
    OAuth2Credentials,
    OAuthState,
    UserMetadata,
    UserMetadataRaw,
)


class SupabaseIntegrationCredentialsStore:
    def __init__(self, supabase: "Client", redis: "Redis"):
        self.supabase = supabase
        self.locks = RedisKeyedMutex(redis)

    def add_creds(self, user_id: str, credentials: Credentials) -> None:
        with self.locked_user_metadata(user_id):
            if self.get_creds_by_id(user_id, credentials.id):
                raise ValueError(
                    f"Can not re-create existing credentials #{credentials.id} "
                    f"for user #{user_id}"
                )
            self._set_user_integration_creds(
                user_id, [*self.get_all_creds(user_id), credentials]
            )

    def get_all_creds(self, user_id: str) -> list[Credentials]:
        user_metadata = self._get_user_metadata(user_id)
        return UserMetadata.model_validate(user_metadata).integration_credentials

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
        with self.locked_user_metadata(user_id):
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
        with self.locked_user_metadata(user_id):
            filtered_credentials = [
                c for c in self.get_all_creds(user_id) if c.id != credentials_id
            ]
            self._set_user_integration_creds(user_id, filtered_credentials)

    async def store_state_token(
        self, user_id: str, provider: str, scopes: list[str]
    ) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        state = OAuthState(
            token=token,
            provider=provider,
            expires_at=int(expires_at.timestamp()),
            scopes=scopes,
        )

        with self.locked_user_metadata(user_id):
            user_metadata = self._get_user_metadata(user_id)
            oauth_states = user_metadata.get("integration_oauth_states", [])
            oauth_states.append(state.model_dump())
            user_metadata["integration_oauth_states"] = oauth_states

            self.supabase.auth.admin.update_user_by_id(
                user_id, {"user_metadata": user_metadata}
            )

        return token

    async def get_any_valid_scopes_from_state_token(
        self, user_id: str, token: str, provider: str
    ) -> list[str]:
        """
        Get the valid scopes from the OAuth state token. This will return any valid scopes
        from any OAuth state token for the given provider. If no valid scopes are found,
        an empty list is returned. DO NOT RELY ON THIS TOKEN TO AUTHENTICATE A USER, AS IT
        IS TO CHECK IF THE USER HAS GIVEN PERMISSIONS TO THE APPLICATION BEFORE EXCHANGING
        THE CODE FOR TOKENS.
        """
        user_metadata = self._get_user_metadata(user_id)
        oauth_states = user_metadata.get("integration_oauth_states", [])

        now = datetime.now(timezone.utc)
        valid_state = next(
            (
                state
                for state in oauth_states
                if state["token"] == token
                and state["provider"] == provider
                and state["expires_at"] > now.timestamp()
            ),
            None,
        )

        if valid_state:
            return valid_state.get("scopes", [])

        return []

    async def verify_state_token(self, user_id: str, token: str, provider: str) -> bool:
        with self.locked_user_metadata(user_id):
            user_metadata = self._get_user_metadata(user_id)
            oauth_states = user_metadata.get("integration_oauth_states", [])

            now = datetime.now(timezone.utc)
            valid_state = next(
                (
                    state
                    for state in oauth_states
                    if state["token"] == token
                    and state["provider"] == provider
                    and state["expires_at"] > now.timestamp()
                ),
                None,
            )

            if valid_state:
                # Remove the used state
                oauth_states.remove(valid_state)
                user_metadata["integration_oauth_states"] = oauth_states
                self.supabase.auth.admin.update_user_by_id(
                    user_id, {"user_metadata": user_metadata}
                )
                return True

        return False

    def _set_user_integration_creds(
        self, user_id: str, credentials: list[Credentials]
    ) -> None:
        raw_metadata = self._get_user_metadata(user_id)
        raw_metadata.update(
            {"integration_credentials": [c.model_dump() for c in credentials]}
        )
        self.supabase.auth.admin.update_user_by_id(
            user_id, {"user_metadata": raw_metadata}
        )

    def _get_user_metadata(self, user_id: str) -> UserMetadataRaw:
        response = self.supabase.auth.admin.get_user_by_id(user_id)
        if not response.user:
            raise ValueError(f"User with ID {user_id} not found")
        return cast(UserMetadataRaw, response.user.user_metadata)

    def locked_user_metadata(self, user_id: str):
        key = (self.supabase.supabase_url, f"user:{user_id}", "metadata")
        return self.locks.locked(key)
