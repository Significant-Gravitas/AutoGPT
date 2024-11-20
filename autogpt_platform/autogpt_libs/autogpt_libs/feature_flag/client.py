import asyncio
import contextlib
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import ldclient
from fastapi import HTTPException, Request
from ldclient import Context, LDClient
from ldclient.config import Config
from typing_extensions import Concatenate, ParamSpec
from .config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

P = ParamSpec("P")
T = TypeVar("T")


def get_client() -> LDClient:
    """Get the LaunchDarkly client singleton."""
    return ldclient.get()


def initialize_launchdarkly() -> None:
    sdk_key = settings.SDK_KEY
    logger.debug(f"Initializing LaunchDarkly with SDK key: {'present' if sdk_key else 'missing'}")

    if not sdk_key:
        logger.warning("LaunchDarkly SDK key not configured")
        return

    config = Config(sdk_key)
    ldclient.set_config(config)

    if ldclient.get().is_initialized():
        logger.info("LaunchDarkly client initialized successfully")
    else:
        logger.error("LaunchDarkly client failed to initialize")

def shutdown_launchdarkly() -> None:
    """Shutdown the LaunchDarkly client."""
    if ldclient.get().is_initialized():
        ldclient.get().close()
        logger.info("LaunchDarkly client closed successfully")

def create_context(user_id: str, additional_attributes: Optional[Dict[str, Any]] = None) -> Context:
    """Create LaunchDarkly context with optional additional attributes."""
    builder = Context.builder(user_id).kind("user")
    if additional_attributes:
        for key, value in additional_attributes.items():
            builder.set(key, value)
    return builder.build()


def feature_flag(
        flag_key: str,
        default: bool = False,
        unauthorized_response: Any = None,
):
    """
    Decorator for feature flag protected endpoints.
    """
    def decorator(
            func: Callable[P, T]
    ) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                user_id = kwargs.get('user_id')
                logger.debug(f"Checking flag '{flag_key}' for user '{user_id}'")

                if not get_client().is_initialized():
                    logger.warning(f"LaunchDarkly not initialized, using default={default}")
                    is_enabled = default
                else:
                    context = create_context(user_id)
                    logger.debug(f"Created context for user: {context}")
                    is_enabled = get_client().variation(flag_key, context, default)
                    logger.debug(f"Flag '{flag_key}' evaluation result: {is_enabled}")

                if not is_enabled:
                    logger.debug(f"Feature '{flag_key}' disabled, returning unauthorized response")
                    if unauthorized_response is not None:
                        return cast(T, unauthorized_response)
                    raise HTTPException(status_code=404, detail="Feature not available")

                logger.debug(f"Feature '{flag_key}' enabled, executing function")
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error evaluating feature flag {flag_key}: {e}", exc_info=True)
                return cast(T, unauthorized_response) if unauthorized_response else None

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                user_id = kwargs.get('user_id')
                if not get_client().is_initialized():
                    logger.warning(f"LaunchDarkly not initialized, using default={default}")
                    is_enabled = default
                else:
                    context = create_context(user_id)
                    is_enabled = get_client().variation(flag_key, context, default)

                if not is_enabled:
                    if unauthorized_response is not None:
                        return cast(T, unauthorized_response)
                    raise HTTPException(status_code=404, detail="Feature not available")

                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error evaluating feature flag {flag_key}: {e}")
                return cast(T, unauthorized_response) if unauthorized_response else None

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextlib.contextmanager
def mock_flag_variation(flag_key: str, return_value: Any):
    """Context manager for testing feature flags."""
    original_variation = get_client().variation
    get_client().variation = lambda key, context, default: return_value if key == flag_key else original_variation(key,
                                                                                                                   context,
                                                                                                                   default)
    try:
        yield
    finally:
        get_client().variation = original_variation


def percentage_rollout(
        flag_key: str,
        default: bool = False,
        unauthorized_response: Any = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for percentage-based rollouts."""
    return feature_flag(flag_key, default, unauthorized_response)


def beta_feature(
        flag_key: str | None = None,
        unauthorized_response: Any = {"message": "Not available in beta"}
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for beta features."""
    actual_key = f"beta-{flag_key}" if flag_key else "beta"
    return feature_flag(actual_key, False, unauthorized_response)