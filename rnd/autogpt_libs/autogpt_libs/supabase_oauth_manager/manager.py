from supabase import create_client, Client
from typing import Dict, Optional
from .types import OAuthTokens, UserMetadata


class SupabaseOAuthManager:
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)

    def update_user_oauth_tokens(self, user_id: str, provider: str, tokens: OAuthTokens) -> Optional[Dict]:
        if not self._is_valid_provider(provider):
            raise ValueError(f"Invalid provider: {provider}")

        if not self._is_valid_token_data(tokens):
            raise ValueError("Invalid token data provided")

        response = self.supabase.auth.admin.update_user_by_id(
            user_id,
            {
                "user_metadata": {
                    "oauth_connections": {
                        provider: tokens
                    }
                }
            }
        )
        if not response.user:
            raise ValueError(f"User with id {user_id} not found")
        return response.user

    def get_user_oauth_tokens(self, user_id: str, provider: str) -> Optional[OAuthTokens]:
        if not self._is_valid_provider(provider):
            raise ValueError(f"Invalid provider: {provider}")

        response = self.supabase.auth.admin.get_user_by_id(user_id)
        if not response.user:
            raise ValueError(f"User with id {user_id} not found")
        return response.user.user_metadata.get("oauth_connections", {}).get(provider)

    def remove_user_oauth_tokens(self, user_id: str, provider: str) -> Optional[Dict]:
        if not self._is_valid_provider(provider):
            raise ValueError(f"Invalid provider: {provider}")

        current_metadata = self.get_user_metadata(user_id)
        if current_metadata and "oauth_connections" in current_metadata:
            current_metadata["oauth_connections"].pop(provider, None)
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

    def _is_valid_provider(self, provider: str) -> bool:
        valid_providers = ['google', 'tiktok']  # Add more as needed
        return provider in valid_providers

    def _is_valid_token_data(self, tokens: OAuthTokens) -> bool:
        required_fields = ['access_token', 'token_type']
        return all(field in tokens for field in required_fields)
