from .base import BaseOAuthHandler
from .github import GitHubOAuthHandler
from .google import GoogleOAuthHandler
from .notion import NotionOAuthHandler

# --8<-- [start:HANDLERS_BY_NAMEExample]
HANDLERS_BY_NAME: dict[str, type[BaseOAuthHandler]] = {
    handler.PROVIDER_NAME: handler
    for handler in [
        GitHubOAuthHandler,
        GoogleOAuthHandler,
        NotionOAuthHandler,
    ]
}
# --8<-- [end:HANDLERS_BY_NAMEExample]

__all__ = ["HANDLERS_BY_NAME"]
