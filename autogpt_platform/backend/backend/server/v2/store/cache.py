"""
FastAPI-specific cache decorator for store endpoints.

This module provides a decorator for caching FastAPI endpoint responses
with configurable TTL and automatic cache key generation.
"""

import functools
import hashlib
import json
import logging
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.util.cache import TTLCache, generate_cache_key
from backend.util.json import loads

logger = logging.getLogger(__name__)

# Create a dedicated cache instance for store endpoints
store_cache = TTLCache(default_ttl=3600, max_size=1000)


def extract_request_params(request: Request) -> Dict[str, Any]:
    """
    Extract all parameters from a FastAPI request.

    Args:
        request: FastAPI Request object

    Returns:
        Dictionary containing all request parameters
    """
    params = {}

    # Include path parameters
    if request.path_params:
        params["path"] = request.path_params

    # Include query parameters
    if request.query_params:
        params["query"] = dict(request.query_params)

    # Include relevant headers (exclude auth and changing headers)
    # We only include headers that affect the response content
    relevant_headers = {}
    for header in ["accept", "accept-language", "content-type"]:
        if header in request.headers:
            relevant_headers[header] = request.headers[header]
    if relevant_headers:
        params["headers"] = relevant_headers

    # Include the path itself
    params["url_path"] = str(request.url.path)

    return params


def generate_endpoint_cache_key(
    func: Callable, request: Request, *args, **kwargs
) -> str:
    """
    Generate a cache key for a FastAPI endpoint.

    Args:
        func: The endpoint function
        request: FastAPI Request object
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

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

    # Combine with function arguments
    all_params = {"request": request_params, "args": args, "kwargs": kwargs}

    # Serialize and hash
    try:
        param_str = json.dumps(all_params, sort_keys=True, default=str)
        hasher.update(param_str.encode())
    except (TypeError, ValueError):
        # Fallback for non-serializable objects
        hasher.update(str(all_params).encode())

    return hasher.hexdigest()


def ttl_cache(
    ttl_seconds: int = 3600,
    skip_auth: bool = True,
    cache_instance: Optional[TTLCache] = None,
) -> Callable:
    """
    Decorator to add TTL caching to FastAPI endpoints.

    Args:
        ttl_seconds: Time-to-live in seconds for cached responses
        skip_auth: Skip caching for authenticated requests
        cache_instance: Custom cache instance (uses global store cache if None)

    Returns:
        Decorated function with caching behavior
    """
    cache = cache_instance or store_cache

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
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

            # Check cache
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
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    # Fall through to execute function

            # Execute function
            logger.info(f"[CACHE MISS] {func.__name__} - fetching from source")
            result = await func(*args, **kwargs)

            # Serialize and cache the result
            try:
                if isinstance(result, (JSONResponse, Response)):
                    # Special handling for Response objects
                    body_content = None
                    if result.body:
                        if isinstance(result.body, bytes):
                            body_content = result.body.decode()
                        elif isinstance(result.body, str):
                            body_content = result.body
                        else:
                            body_content = str(result.body)
                    cached_value = {
                        "__response_type__": result.__class__.__name__,
                        "content": body_content,
                        "headers": dict(result.headers),
                        "status_code": result.status_code,
                    }
                elif hasattr(result, "model_dump"):
                    # Pydantic model
                    cached_value = result.model_dump()
                else:
                    # Regular object
                    cached_value = result

                cache.set(cache_key, cached_value, ttl_seconds)
                logger.info(f"[CACHE SET] {func.__name__} - cached for {ttl_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to cache result for {func.__name__}: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync endpoints
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get("request")

            if skip_auth and request:
                auth_header = request.headers.get("authorization")
                user_id = kwargs.get("user_id")
                if auth_header or user_id:
                    logger.debug(
                        f"Skipping cache for authenticated request to {func.__name__}"
                    )
                    return func(*args, **kwargs)

            if request:
                cache_key = generate_endpoint_cache_key(func, request, *args, **kwargs)
            else:
                cache_key = generate_cache_key(func, args, kwargs)

            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"[CACHE HIT] {func.__name__} - serving from cache")

                try:
                    if (
                        isinstance(cached_result, dict)
                        and "__response_type__" in cached_result
                    ):
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
                    return cached_result
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")

            logger.info(f"[CACHE MISS] {func.__name__} - fetching from source")
            result = func(*args, **kwargs)

            try:
                if isinstance(result, (JSONResponse, Response)):
                    body_content = None
                    if result.body:
                        if isinstance(result.body, bytes):
                            body_content = result.body.decode()
                        elif isinstance(result.body, str):
                            body_content = result.body
                        else:
                            body_content = str(result.body)
                    cached_value = {
                        "__response_type__": result.__class__.__name__,
                        "content": body_content,
                        "headers": dict(result.headers),
                        "status_code": result.status_code,
                    }
                elif hasattr(result, "model_dump"):
                    cached_value = result.model_dump()
                else:
                    cached_value = result

                cache.set(cache_key, cached_value, ttl_seconds)
                logger.info(f"[CACHE SET] {func.__name__} - cached for {ttl_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to cache result for {func.__name__}: {e}")

            return result

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def clear_store_cache():
    """Clear all entries from the store cache."""
    store_cache.clear()
    logger.info("Store cache cleared")


def get_store_cache_stats() -> Dict[str, Any]:
    """Get statistics about the store cache."""
    return store_cache.stats()
