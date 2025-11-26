from datetime import datetime
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Union
from urllib.parse import urlencode

from pydantic import field_serializer

from backend.sdk import BaseModel, Credentials, Requests

logger = getLogger(__name__)

WORDPRESS_BASE_URL = "https://public-api.wordpress.com/"


class OAuthAuthorizeRequest(BaseModel):
    """OAuth authorization request parameters for WordPress.

    Parameters:
        client_id: Your application's client ID from WordPress.com
        redirect_uri: The URI for the authorize response redirect. Must exactly match a redirect URI
            associated with your application.
        response_type: Can be "code" or "token". "code" should be used for server side applications.
        scope: A space delimited list of scopes. Optional, defaults to single blog access.
        blog: Optional blog parameter with the URL or blog ID for a WordPress.com blog or Jetpack site.
    """

    client_id: str
    redirect_uri: str
    response_type: str = "code"
    scope: str | None = None
    blog: str | None = None


class OAuthTokenRequest(BaseModel):
    """OAuth token request parameters for WordPress.

    These parameters must be formatted via application/x-www-form-urlencoded encoding.

    Parameters:
        code: The grant code returned in the redirect. Can only be used once.
        client_id: Your application's client ID.
        redirect_uri: The redirect_uri used in the authorization request.
        client_secret: Your application's client secret.
        grant_type: The string "authorization_code".
    """

    code: str
    client_id: str
    redirect_uri: str
    client_secret: str
    grant_type: str = "authorization_code"


class OAuthRefreshTokenRequest(BaseModel):
    """OAuth token refresh request parameters for WordPress.

    Note: WordPress OAuth2 tokens do not expire when using the "code" response type,
    so refresh tokens are typically not needed for server-side applications.

    Parameters:
        refresh_token: The saved refresh token from the previous token grant.
        client_id: Your application's client ID.
        client_secret: Your application's client secret.
        grant_type: The string "refresh_token".
    """

    refresh_token: str
    client_id: str
    client_secret: str
    grant_type: str = "refresh_token"


class OAuthTokenResponse(BaseModel):
    """OAuth token response from WordPress.

    Successful response has HTTP status code 200 (OK).

    Parameters:
        access_token: An opaque string. Can be used to make requests to the WordPress API on behalf
            of the user.
        blog_id: The ID of the authorized blog.
        blog_url: The URL of the authorized blog.
        token_type: The string "bearer".
        scope: Optional field for global tokens containing the granted scopes.
        refresh_token: Optional refresh token (typically not provided for server-side apps).
        expires_in: Optional expiration time (tokens from code flow don't expire).
    """

    access_token: str
    blog_id: str | None = None
    blog_url: str | None = None
    token_type: str = "bearer"
    scope: str | None = None
    refresh_token: str | None = None
    expires_in: int | None = None


def make_oauth_authorize_url(
    client_id: str,
    redirect_uri: str,
    scopes: list[str] | None = None,
) -> str:
    """
    Generate the OAuth authorization URL for WordPress.

    Args:
        client_id: Your application's client ID from WordPress.com
        redirect_uri: The URI for the authorize response redirect
        scopes: Optional list of scopes. Defaults to single blog access if not provided.
        blog: Optional blog URL or ID for a WordPress.com blog or Jetpack site.

    Returns:
        The authorization URL that the user should visit
    """
    # Build request parameters
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    }

    if scopes:
        params["scope"] = " ".join(scopes)

    # Build the authorization URL
    base_url = f"{WORDPRESS_BASE_URL}oauth2/authorize"
    query_string = urlencode(params)

    return f"{base_url}?{query_string}"


async def oauth_exchange_code_for_tokens(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> OAuthTokenResponse:
    """
    Exchange an authorization code for access token.

    Args:
        client_id: Your application's client ID.
        client_secret: Your application's client secret.
        code: The authorization code returned by WordPress.
        redirect_uri: The redirect URI used during authorization.

    Returns:
        Parsed JSON response containing the access token, blog info, etc.
    """

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = OAuthTokenRequest(
        code=code,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        grant_type="authorization_code",
    ).model_dump(exclude_none=True)

    response = await Requests().post(
        f"{WORDPRESS_BASE_URL}oauth2/token",
        headers=headers,
        data=data,
    )

    if response.ok:
        return OAuthTokenResponse.model_validate(response.json())
    raise ValueError(
        f"Failed to exchange code for tokens: {response.status} {response.text}"
    )


async def oauth_refresh_tokens(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> OAuthTokenResponse:
    """
    Refresh an expired access token (for implicit/client-side tokens only).

    Note: Tokens obtained via the "code" flow for server-side applications do not expire.
    This is primarily used for client-side applications using implicit OAuth.

    Args:
        client_id: Your application's client ID.
        client_secret: Your application's client secret.
        refresh_token: The refresh token previously issued by WordPress.

    Returns:
        Parsed JSON response containing the new access token.
    """

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = OAuthRefreshTokenRequest(
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        grant_type="refresh_token",
    ).model_dump(exclude_none=True)

    response = await Requests().post(
        f"{WORDPRESS_BASE_URL}oauth2/token",
        headers=headers,
        data=data,
    )

    if response.ok:
        return OAuthTokenResponse.model_validate(response.json())
    raise ValueError(f"Failed to refresh tokens: {response.status} {response.text}")


class TokenInfoResponse(BaseModel):
    """Token validation response from WordPress.

    Parameters:
        client_id: Your application's client ID.
        user_id: The WordPress.com user ID.
        blog_id: The blog ID associated with the token.
        scope: The scope of the token.
    """

    client_id: str
    user_id: str
    blog_id: str | None = None
    scope: str | None = None


async def validate_token(
    client_id: str,
    token: str,
) -> TokenInfoResponse:
    """
    Validate an access token and get associated metadata.

    Args:
        client_id: Your application's client ID.
        token: The access token to validate.

    Returns:
        Token info including user ID, blog ID, and scope.
    """

    params = {
        "client_id": client_id,
        "token": token,
    }

    response = await Requests().get(
        f"{WORDPRESS_BASE_URL}oauth2/token-info",
        params=params,
    )

    if response.ok:
        return TokenInfoResponse.model_validate(response.json())
    raise ValueError(f"Invalid token: {response.status} {response.text}")


async def make_api_request(
    endpoint: str,
    access_token: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Make an authenticated API request to WordPress.

    Args:
        endpoint: The API endpoint (e.g., "/rest/v1/me/", "/rest/v1/sites/{site_id}/posts/new")
        access_token: The OAuth access token
        method: HTTP method (GET, POST, etc.)
        data: Request body data for POST/PUT requests
        params: Query parameters

    Returns:
        JSON response from the API
    """

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    if data and method in ["POST", "PUT", "PATCH"]:
        headers["Content-Type"] = "application/json"

    # Ensure endpoint starts with /
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    url = f"{WORDPRESS_BASE_URL.rstrip('/')}{endpoint}"

    request_method = getattr(Requests(), method.lower())
    response = await request_method(
        url,
        headers=headers,
        json=data if method in ["POST", "PUT", "PATCH"] else None,
        params=params,
    )

    if response.ok:
        return response.json()
    raise ValueError(f"API request failed: {response.status} {response.text}")


# Post-related models and functions


class PostStatus(str, Enum):
    """WordPress post status options."""

    PUBLISH = "publish"
    PRIVATE = "private"
    DRAFT = "draft"
    PENDING = "pending"
    FUTURE = "future"
    AUTO_DRAFT = "auto-draft"


class PostFormat(str, Enum):
    """WordPress post format options."""

    STANDARD = "standard"
    ASIDE = "aside"
    CHAT = "chat"
    GALLERY = "gallery"
    LINK = "link"
    IMAGE = "image"
    QUOTE = "quote"
    STATUS = "status"
    VIDEO = "video"
    AUDIO = "audio"


class CreatePostRequest(BaseModel):
    """Request model for creating a WordPress post.

    All fields are optional except those you want to set.
    """

    # Basic content
    title: str | None = None
    content: str | None = None
    excerpt: str | None = None

    # Post metadata
    date: datetime | None = None
    slug: str | None = None
    author: str | None = None
    status: PostStatus | None = PostStatus.PUBLISH
    password: str | None = None
    sticky: bool | None = False

    # Organization
    parent: int | None = None
    type: str | None = "post"
    categories: List[str] | None = None
    tags: List[str] | None = None
    format: PostFormat | None = None

    # Media
    featured_image: str | None = None
    media_urls: List[str] | None = None

    # Sharing
    publicize: bool | None = True
    publicize_message: str | None = None

    # Engagement
    likes_enabled: bool | None = None
    sharing_enabled: bool | None = True
    discussion: Dict[str, bool] | None = None

    # Page-specific
    menu_order: int | None = None
    page_template: str | None = None

    # Advanced
    metadata: List[Dict[str, Any]] | None = None

    @field_serializer("date")
    def serialize_date(self, value: datetime | None) -> str | None:
        return value.isoformat() if value else None


class PostAuthor(BaseModel):
    """Author information in post response."""

    ID: int
    login: str
    email: Union[str, bool, None] = None
    name: str
    nice_name: str
    URL: str | None = None
    avatar_URL: str | None = None


class PostResponse(BaseModel):
    """Response model for a WordPress post."""

    ID: int
    site_ID: int
    author: PostAuthor
    date: datetime
    modified: datetime
    title: str
    URL: str
    short_URL: str
    content: str
    excerpt: str
    slug: str
    guid: str
    status: str
    sticky: bool
    password: str | None = ""
    parent: Union[Dict[str, Any], bool, None] = None
    type: str
    discussion: Dict[str, Union[str, bool, int]]
    likes_enabled: bool
    sharing_enabled: bool
    like_count: int
    i_like: bool
    is_reblogged: bool
    is_following: bool
    global_ID: str
    featured_image: str | None = None
    post_thumbnail: Dict[str, Any] | None = None
    format: str
    geo: Union[Dict[str, Any], bool, None] = None
    menu_order: int | None = None
    page_template: str | None = None
    publicize_URLs: List[str]
    terms: Dict[str, Dict[str, Any]]
    tags: Dict[str, Dict[str, Any]]
    categories: Dict[str, Dict[str, Any]]
    attachments: Dict[str, Dict[str, Any]]
    attachment_count: int
    metadata: List[Dict[str, Any]]
    meta: Dict[str, Any]
    capabilities: Dict[str, bool]
    revisions: List[int] | None = None
    other_URLs: Dict[str, Any] | None = None


async def create_post(
    credentials: Credentials,
    site: str,
    post_data: CreatePostRequest,
) -> PostResponse:
    """
    Create a new post on a WordPress site.

    Args:
        site: Site ID or domain (e.g., "myblog.wordpress.com" or "123456789")
        access_token: OAuth access token
        post_data: Post data using CreatePostRequest model

    Returns:
        PostResponse with the created post details
    """

    # Convert the post data to a dictionary, excluding None values
    data = post_data.model_dump(exclude_none=True)

    # Handle special fields that need conversion
    if "categories" in data and isinstance(data["categories"], list):
        data["categories"] = ",".join(str(c) for c in data["categories"])

    if "tags" in data and isinstance(data["tags"], list):
        data["tags"] = ",".join(str(t) for t in data["tags"])

    # Make the API request
    endpoint = f"/rest/v1.1/sites/{site}/posts/new"

    headers = {
        "Authorization": credentials.auth_header(),
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = await Requests().post(
        f"{WORDPRESS_BASE_URL.rstrip('/')}{endpoint}",
        headers=headers,
        data=data,
    )

    if response.ok:
        return PostResponse.model_validate(response.json())

    error_data = (
        response.json()
        if response.headers.get("content-type", "").startswith("application/json")
        else {}
    )
    error_message = error_data.get("message", response.text)
    raise ValueError(f"Failed to create post: {response.status} - {error_message}")
