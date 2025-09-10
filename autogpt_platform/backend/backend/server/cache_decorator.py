"""
FastAPI cache decorator for adding TTL caching to endpoints.

This module provides a decorator that can be applied to FastAPI endpoints
to cache their responses with configurable time-to-live (TTL) settings.
"""

import asyncio
import functools
import hashlib
import json
import logging
from json import loads
from typing import Any, Callable, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from backend.server.cache_manager import CacheComponent, get_component_cache
from backend.util.cache import TTLCache, generate_cache_key

logger = logging.getLogger(__name__)


def extract_request_params(request: Request) -> Dict[str, Any]:
    """
    Extract all parameters from a FastAPI request.

    Args:
        request: The FastAPI Request object

    Returns:
        Dictionary containing query params, path params, and headers
    """
    params = {}

    # Extract query parameters
    if request.query_params:
        params["query"] = dict(request.query_params)

    # Extract path parameters
    if hasattr(request, "path_params") and request.path_params:
        params["path"] = dict(request.path_params)

    # Extract relevant headers (exclude sensitive ones)
    safe_headers = ["accept", "content-type", "accept-language"]
    headers = {}
    for header in safe_headers:
        if header in request.headers:
            headers[header] = request.headers[header]
    if headers:
        params["headers"] = headers

    # Include request method and path
    params["method"] = request.method
    params["path"] = request.url.path

    return params


def generate_endpoint_cache_key(
    func: Callable, request: Request, *args, **kwargs
) -> str:
    """
    Generate a cache key for a FastAPI endpoint.

    Args:
        func: The endpoint function
        request: The FastAPI Request object
        args: Positional arguments to the function
        kwargs: Keyword arguments to the function

    Returns:
        SHA256 hash string as cache key
    """
    hasher = hashlib.sha256()

    # Include function identity
    hasher.update(func.__module__.encode())
    hasher.update(func.__name__.encode())

    # Include function source code to invalidate on changes
    try:
        import inspect

        source = inspect.getsource(func)
        hasher.update(source.encode())
    except (OSError, TypeError):
        pass

    # Extract and include request parameters
    request_params = extract_request_params(request)

    # If user_id is in kwargs, include it prominently in the key
    # This makes user-specific invalidation easier
    user_id = kwargs.get("user_id")
    if user_id:
        hasher.update(f"user:{user_id}".encode())

    # Combine with function arguments
    all_params = {"request": request_params, "args": args, "kwargs": kwargs}

    # Serialize and hash
    try:
        param_str = json.dumps(all_params, sort_keys=True, default=str)
        hasher.update(param_str.encode())
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        hasher.update(str(all_params).encode())

    # Create a more readable cache key prefix for logging
    key_prefix = f"{func.__module__}.{func.__name__}"
    if user_id:
        key_prefix = f"{key_prefix}:user:{user_id}"

    hash_suffix = hasher.hexdigest()
    return f"{key_prefix}:{hash_suffix[:16]}"


def ttl_cache(
    ttl_seconds: int = 3600,
    skip_auth: bool = True,
    cache_instance: Optional[TTLCache] = None,
    cache_component: Optional[CacheComponent] = None,
    include_user_id: bool = True,
) -> Callable:
    """
    Decorator to add TTL caching to FastAPI endpoints.

    Args:
        ttl_seconds: Time-to-live in seconds for cached responses
        skip_auth: Skip caching for authenticated requests
        cache_instance: Custom cache instance (uses component cache if None)
        cache_component: Cache component to use (if cache_instance not provided)
        include_user_id: Whether to include user_id in cache key

    Returns:
        Decorated function with caching behavior
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Lazy cache resolution to avoid initialization issues
            if cache_instance:
                cache = cache_instance
            elif cache_component:
                cache = get_component_cache(cache_component)
            else:
                # Default to V1_API cache if no component specified
                cache = get_component_cache(CacheComponent.V1_API)

            # If caching is disabled, call the original function
            if not cache:
                return await func(*args, **kwargs)

            # Try to find Request object in kwargs or args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get("request")

            # Skip cache for authenticated requests if configured
            if skip_auth and request:
                auth_header = request.headers.get("authorization")
                user_id = kwargs.get("user_id")
                if auth_header or user_id:
                    logger.debug(
                        f"Skipping cache for authenticated request to {func.__name__}"
                    )
                    return await func(*args, **kwargs)

            # Generate cache key
            if request:
                cache_key = generate_endpoint_cache_key(func, request, *args, **kwargs)
            else:
                # Fallback for non-request endpoints
                cache_key = generate_cache_key(func, args, kwargs)
            
            logger.debug(f"[CACHE] Generated cache key for {func.__name__}: {cache_key}")

            # Check cache (cache is guaranteed to be non-None here)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"[CACHE HIT] {func.__name__} - serving from cache")

                # Deserialize the cached response
                try:
                    if (
                        isinstance(cached_result, dict)
                        and "__response_type__" in cached_result
                    ):
                        # Restore the original response type
                        response_type = cached_result["__response_type__"]
                        content = cached_result["content"]
                        headers = cached_result.get("headers", {})
                        status_code = cached_result.get("status_code", 200)

                        if response_type == "JSONResponse":
                            return JSONResponse(
                                content=(
                                    loads(content)
                                    if isinstance(content, str)
                                    else content
                                ),
                                headers=headers,
                                status_code=status_code,
                            )
                        elif response_type == "Response":
                            return Response(
                                content=content,
                                headers=headers,
                                status_code=status_code,
                            )
                    # Return as-is for regular objects
                    return cached_result
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize cached response for {func.__name__}: {e}"
                    )
                    # Fall through to execute the function

            # Execute the function
            logger.debug(f"[CACHE MISS] {func.__name__} - executing function")
            result = await func(*args, **kwargs)

            # Serialize response objects for caching
            cache_value = result
            if isinstance(result, (Response, JSONResponse)):
                # For Response objects, we need to preserve the response type
                # and extract the content
                cache_value = {
                    "__response_type__": type(result).__name__,
                    "status_code": result.status_code,
                    "headers": dict(result.headers) if result.headers else {},
                }

                # Handle body content
                if hasattr(result, "body"):
                    body = result.body
                    # Check if body is already bytes or string
                    if isinstance(body, (bytes, str)):
                        cache_value["content"] = (
                            body.decode("utf-8") if isinstance(body, bytes) else body
                        )
                    else:
                        # For other types, convert to JSON string
                        cache_value["content"] = json.dumps(body, default=str)

            # Store in cache
            cache.set(cache_key, cache_value, ttl=ttl_seconds)
            logger.debug(
                f"[CACHE SET] {func.__name__} - cached with key {cache_key[:50]}... for {ttl_seconds} seconds"
            )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions (shouldn't happen in FastAPI but just in case)
            logger.warning(
                f"Cache decorator applied to sync function {func.__name__}, "
                "caching disabled"
            )
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
