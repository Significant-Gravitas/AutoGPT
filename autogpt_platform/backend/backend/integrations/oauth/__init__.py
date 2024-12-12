from typing import TYPE_CHECKING

from .github import GitHubOAuthHandler
from .google import GoogleOAuthHandler
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
    ]
}
# --8<-- [end:HANDLERS_BY_NAMEExample]

__all__ = ["HANDLERS_BY_NAME"]
