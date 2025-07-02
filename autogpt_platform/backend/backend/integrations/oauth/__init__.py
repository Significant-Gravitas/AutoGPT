from typing import TYPE_CHECKING

from backend.integrations.oauth.todoist import TodoistOAuthHandler

from .github import GitHubOAuthHandler
from .google import GoogleOAuthHandler
from .notion import NotionOAuthHandler
from .twitter import TwitterOAuthHandler

if TYPE_CHECKING:
    from .base import BaseOAuthHandler

# --8<-- [start:HANDLERS_BY_NAMEExample]
# Build handlers dict with string keys for compatibility with SDK auto-registration
_ORIGINAL_HANDLERS = [
    GitHubOAuthHandler,
    GoogleOAuthHandler,
    NotionOAuthHandler,
    TwitterOAuthHandler,
    TodoistOAuthHandler,
]

# Start with original handlers
_handlers_dict = {
    (
        handler.PROVIDER_NAME.value
        if hasattr(handler.PROVIDER_NAME, "value")
        else str(handler.PROVIDER_NAME)
    ): handler
    for handler in _ORIGINAL_HANDLERS
}


# Create a custom dict class that includes SDK handlers
class SDKAwareHandlersDict(dict):
    """Dictionary that automatically includes SDK-registered OAuth handlers."""

    def __getitem__(self, key):
        # First try the original handlers
        if key in _handlers_dict:
            return _handlers_dict[key]

        # Then try SDK handlers
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            if key in sdk_handlers:
                return sdk_handlers[key]
        except ImportError:
            pass

        # If not found, raise KeyError
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if key in _handlers_dict:
            return True
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            return key in sdk_handlers
        except ImportError:
            return False

    def keys(self):
        # Combine all keys into a single dict and return its keys view
        combined = dict(_handlers_dict)
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            combined.update(sdk_handlers)
        except ImportError:
            pass
        return combined.keys()

    def values(self):
        combined = dict(_handlers_dict)
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            combined.update(sdk_handlers)
        except ImportError:
            pass
        return combined.values()

    def items(self):
        combined = dict(_handlers_dict)
        try:
            from backend.sdk import AutoRegistry

            sdk_handlers = AutoRegistry.get_oauth_handlers()
            combined.update(sdk_handlers)
        except ImportError:
            pass
        return combined.items()


HANDLERS_BY_NAME: dict[str, type["BaseOAuthHandler"]] = SDKAwareHandlersDict()
# --8<-- [end:HANDLERS_BY_NAMEExample]

__all__ = ["HANDLERS_BY_NAME"]
