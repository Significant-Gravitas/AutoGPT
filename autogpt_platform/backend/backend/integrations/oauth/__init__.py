from .base import BaseOAuthHandler
from .github import GitHubOAuthHandler
from .google import GoogleOAuthHandler
from .notion import NotionOAuthHandler

HANDLERS_BY_NAME: dict[str, type[BaseOAuthHandler]] = {
    handler.PROVIDER_NAME: handler
    for handler in [
        GitHubOAuthHandler,
        GoogleOAuthHandler,
        NotionOAuthHandler,
    ]
}

__all__ = ["HANDLERS_BY_NAME"]
