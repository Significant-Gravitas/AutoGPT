from typing import TypedDict, List, Optional


class OAuthTokens(TypedDict):
    access_token: str
    refresh_token: Optional[str]
    token_type: str
    expires_at: Optional[int]
    scopes: List[str]


class UserOAuthConnections(TypedDict):
    google: Optional[OAuthTokens]
    tiktok: Optional[OAuthTokens]


class UserMetadata(TypedDict):
    oauth_connections: UserOAuthConnections
