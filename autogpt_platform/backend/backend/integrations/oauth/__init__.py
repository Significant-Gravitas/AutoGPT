from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

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


class SDKAwareCredentials(BaseModel):
    """OAuth credentials configuration."""

    use_secrets: bool = True
    client_id_env_var: Optional[str] = None
    client_secret_env_var: Optional[str] = None


_credentials_by_provider = {}
# Add default credentials for original handlers
for handler in _ORIGINAL_HANDLERS:
    provider_name = (
        handler.PROVIDER_NAME.value
        if hasattr(handler.PROVIDER_NAME, "value")
        else str(handler.PROVIDER_NAME)
    )
    _credentials_by_provider[provider_name] = SDKAwareCredentials(
        use_secrets=True, client_id_env_var=None, client_secret_env_var=None
    )


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


class SDKAwareCredentialsDict(dict):
    """Dictionary that automatically includes SDK-registered OAuth credentials."""

    def __getitem__(self, key):
        # First try the original handlers
        if key in _credentials_by_provider:
            return _credentials_by_provider[key]

        # Then try SDK credentials
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            if key in sdk_credentials:
                # Convert from SDKOAuthCredentials to SDKAwareCredentials
                sdk_cred = sdk_credentials[key]
                return SDKAwareCredentials(
                    use_secrets=sdk_cred.use_secrets,
                    client_id_env_var=sdk_cred.client_id_env_var,
                    client_secret_env_var=sdk_cred.client_secret_env_var,
                )
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
        if key in _credentials_by_provider:
            return True
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            return key in sdk_credentials
        except ImportError:
            return False

    def keys(self):
        # Combine all keys into a single dict and return its keys view
        combined = dict(_credentials_by_provider)
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            combined.update(sdk_credentials)
        except ImportError:
            pass
        return combined.keys()

    def values(self):
        combined = dict(_credentials_by_provider)
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            # Convert SDK credentials to SDKAwareCredentials
            for key, sdk_cred in sdk_credentials.items():
                combined[key] = SDKAwareCredentials(
                    use_secrets=sdk_cred.use_secrets,
                    client_id_env_var=sdk_cred.client_id_env_var,
                    client_secret_env_var=sdk_cred.client_secret_env_var,
                )
        except ImportError:
            pass
        return combined.values()

    def items(self):
        combined = dict(_credentials_by_provider)
        try:
            from backend.sdk import AutoRegistry

            sdk_credentials = AutoRegistry.get_oauth_credentials()
            # Convert SDK credentials to SDKAwareCredentials
            for key, sdk_cred in sdk_credentials.items():
                combined[key] = SDKAwareCredentials(
                    use_secrets=sdk_cred.use_secrets,
                    client_id_env_var=sdk_cred.client_id_env_var,
                    client_secret_env_var=sdk_cred.client_secret_env_var,
                )
        except ImportError:
            pass
        return combined.items()


HANDLERS_BY_NAME: dict[str, type["BaseOAuthHandler"]] = SDKAwareHandlersDict()
CREDENTIALS_BY_PROVIDER: dict[str, SDKAwareCredentials] = SDKAwareCredentialsDict()
# --8<-- [end:HANDLERS_BY_NAMEExample]

__all__ = ["HANDLERS_BY_NAME"]
