from typing import TYPE_CHECKING

from backend.integrations.oauth.todoist import TodoistOAuthHandler

from .github import GitHubOAuthHandler
from .google import GoogleOAuthHandler
from .linear import LinearOAuthHandler
from .notion import NotionOAuthHandler
from .twitter import TwitterOAuthHandler

if TYPE_CHECKING:
    from ..providers import ProviderName
    from .base import BaseOAuthHandler

# --8<-- [start:HANDLERS_BY_NAMEExample]
HANDLERS_BY_NAME: dict["ProviderName", type["BaseOAuthHandler"]] = {
    handler.PROVIDER_NAME: handler
    for handler in [
        GitHubOAuthHandler,
        GoogleOAuthHandler,
        NotionOAuthHandler,
        TwitterOAuthHandler,
        LinearOAuthHandler,
        TodoistOAuthHandler,
    ]
}
# --8<-- [end:HANDLERS_BY_NAMEExample]

__all__ = ["HANDLERS_BY_NAME"]
