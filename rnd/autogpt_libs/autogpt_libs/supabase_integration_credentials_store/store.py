from typing import cast

from supabase import Client, create_client

from .types import Credentials, OAuth2Credentials, UserMetadata, UserMetadataRaw


class SupabaseIntegrationCredentialsStore:
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)

    def add_creds(self, user_id: str, credentials: Credentials) -> None:
        if self.get_creds_by_id(user_id, credentials.id):
            raise ValueError(
                f"Can not re-create existing credentials with ID {credentials.id} "
                f"for user with ID {user_id}"
            )
        self._set_user_integration_creds(
            user_id, [*self.get_all_creds(user_id), credentials]
        )

    def get_all_creds(self, user_id: str) -> list[Credentials]:
        user_metadata = self._get_user_metadata(user_id)
        return UserMetadata.model_validate(user_metadata).integration_credentials

    def get_creds_by_id(self, user_id: str, credentials_id: str) -> Credentials | None:
        credentials = self.get_all_creds(user_id)
        return next((c for c in credentials if c.id == credentials_id), None)

    def get_creds_by_provider(self, user_id: str, provider: str) -> list[Credentials]:
        credentials = self.get_all_creds(user_id)
        return [c for c in credentials if c.provider == provider]

    def get_authorized_providers(self, user_id: str) -> list[str]:
        credentials = self.get_all_creds(user_id)
        return list(set(c.provider for c in credentials))

    def update_creds(self, user_id: str, updated: Credentials) -> None:
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
            updated if c.id == updated.id else c for c in self.get_all_creds(user_id)
        ]
        self._set_user_integration_creds(user_id, updated_credentials_list)

    def delete_creds_by_id(self, user_id: str, credentials_id: str) -> None:
        filtered_credentials = [
            c for c in self.get_all_creds(user_id) if c.id != credentials_id
        ]
        self._set_user_integration_creds(user_id, filtered_credentials)

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
