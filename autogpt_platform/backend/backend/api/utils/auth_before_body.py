"""Check a signed-in route's token before FastAPI parses the request body.

FastAPI reads and parses a route's body before it resolves the route's
dependencies, the sign-in check among them. Without this, a request with no
token or a bad one still had its upload parsed and spooled to disk, or its
JSON parsed, before it got its 401.
"""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from autogpt_libs.auth.jwt_utils import bearer_jwt_auth, get_jwt_payload
from fastapi import FastAPI, Request, Response
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_dependant
from fastapi.routing import APIRoute, request_response
from starlette.requests import ClientDisconnect


def check_token_before_body(app: FastAPI) -> None:
    """Make every route of `app` that needs a signed-in user check the token
    before it reads a request body. Call it once, after the last route."""
    for route in app.routes:
        if isinstance(route, APIRoute) and _needs_token(route.dependant, {}):
            route.app = request_response(_token_first(route))


def _token_first(route: APIRoute) -> Callable[[Request], Awaitable[Response]]:
    handle = route.get_route_handler()

    async def check_then_handle(request: Request) -> Response:
        overrides = _overrides(route)
        if _has_body(request) and (
            not overrides or _needs_token(route.dependant, overrides)
        ):
            await _check_token(request)
        return await handle(request)

    return check_then_handle


async def _check_token(request: Request) -> None:
    """Raise the 401 the route's own dependency would, before the body is parsed.

    A refused request's body is still received, chunk by chunk, and thrown
    away, so the client reads the 401 whatever it does with the connection (a
    browser with an expired token refreshes it and retries). Answering with
    the body unread can reset the connection before the client reads it.

    A valid token is verified again when the route resolves its dependencies,
    which go on to check bans, organizations and impersonation. That second
    check is deliberate: this one only keeps a refused request from parsing
    or spooling its body, it doesn't replace the route's own."""
    try:
        await get_jwt_payload(await bearer_jwt_auth(request))
    except Exception:
        # Refused, or failed unexpectedly: either way the body is thrown away
        # before the error goes out.
        try:
            async for _ in request.stream():
                pass
        except ClientDisconnect:
            pass  # Gone mid-upload: still answered as refused, not as a crash.
        raise


def _needs_token(dependant: Dependant, overrides: Mapping[Any, Any]) -> bool:
    """Whether resolving `dependant` reaches `get_jwt_payload`, as FastAPI
    resolves it: an overridden dependency is replaced by its override's own
    dependency tree, built the way `solve_dependencies` builds it.

    `get_jwt_payload` is the anchor: a route that authenticates some other way
    isn't checked early until that way is added here. The early check runs the
    real `get_jwt_payload` and bearer scheme, so when something below it is
    overridden, the route is left to FastAPI."""
    for sub in dependant.dependencies:
        if sub.call in overrides:
            sub = get_dependant(
                path=sub.path or "",
                call=overrides[sub.call],
                name=sub.name,
                parent_oauth_scopes=sub.oauth_scopes,
                scope=sub.scope,
            )
        if sub.call is get_jwt_payload:
            if not any(d.call in overrides for d in sub.dependencies):
                return True
        elif _needs_token(sub, overrides):
            return True
    return False


def _overrides(route: APIRoute) -> Mapping[Any, Any]:
    provider = route.dependency_overrides_provider
    return provider.dependency_overrides if provider is not None else {}


def _has_body(request: Request) -> bool:
    headers = request.headers
    return "transfer-encoding" in headers or headers.get("content-length", "0") != "0"
