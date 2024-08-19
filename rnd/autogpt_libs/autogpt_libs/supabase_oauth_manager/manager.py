from supabase import create_client, Client
from typing import Dict, Optional, List
from .types import OAuthTokens, UserMetadata


class SupabaseOAuthManager:
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)

    def update_user_oauth_tokens(self, user_id: str, provider: str, tokens: OAuthTokens) -> Optional[Dict]:
        if not self._is_valid_token_data(tokens):
            raise ValueError("Invalid token data provided")

        current_metadata = self.get_user_metadata(user_id)

        if current_metadata is None:
            current_metadata = {}

        if 'oauth_connections' not in current_metadata:
            current_metadata['oauth_connections'] = {}

        current_metadata['oauth_connections'][provider] = tokens

        return self.update_user_metadata(user_id, current_metadata)

    def get_user_oauth_tokens(self, user_id: str, provider: str) -> Optional[OAuthTokens]:
        current_metadata = self.get_user_metadata(user_id)
        if current_metadata and 'oauth_connections' in current_metadata:
            return current_metadata['oauth_connections'].get(provider)
        return None

    def remove_user_oauth_tokens(self, user_id: str, provider: str) -> Optional[Dict]:
        current_metadata = self.get_user_metadata(user_id)
        if current_metadata and 'oauth_connections' in current_metadata:
            if provider in current_metadata['oauth_connections']:
                del current_metadata['oauth_connections'][provider]
                return self.update_user_metadata(user_id, current_metadata)
        return None

    def get_user_metadata(self, user_id: str) -> Optional[UserMetadata]:
        response = self.supabase.auth.admin.get_user_by_id(user_id)
        if not response.user:
            raise ValueError(f"User with id {user_id} not found")
        return response.user.user_metadata

    def update_user_metadata(self, user_id: str, metadata: UserMetadata) -> Optional[Dict]:
        response = self.supabase.auth.admin.update_user_by_id(
            user_id,
            {"user_metadata": metadata}
        )
        if not response.user:
            raise ValueError(f"User with id {user_id} not found")
        return response.user

    def get_authorized_providers(self, user_id: str) -> List[str]:
        current_metadata = self.get_user_metadata(user_id)
        if current_metadata and 'oauth_connections' in current_metadata:
            return list(current_metadata['oauth_connections'].keys())
        return []

    def _is_valid_token_data(self, tokens: OAuthTokens) -> bool:
        required_fields = ['access_token', 'token_type']
        return all(field in tokens for field in required_fields)
