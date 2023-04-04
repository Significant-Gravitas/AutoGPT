import re
from typing import TYPE_CHECKING, Awaitable, Callable, Tuple, Type, TypeVar

from .typedefs import Handler
from .web_exceptions import HTTPPermanentRedirect, _HTTPMove
from .web_request import Request
from .web_response import StreamResponse
from .web_urldispatcher import SystemRoute

__all__ = (
    "middleware",
    "normalize_path_middleware",
)

if TYPE_CHECKING:  # pragma: no cover
    from .web_app import Application

_Func = TypeVar("_Func")


async def _check_request_resolves(request: Request, path: str) -> Tuple[bool, Request]:
    alt_request = request.clone(rel_url=path)

    match_info = await request.app.router.resolve(alt_request)
    alt_request._match_info = match_info

    if match_info.http_exception is None:
        return True, alt_request

    return False, request


def middleware(f: _Func) -> _Func:
    f.__middleware_version__ = 1  # type: ignore[attr-defined]
    return f


_Middleware = Callable[[Request, Handler], Awaitable[StreamResponse]]


def normalize_path_middleware(
    *,
    append_slash: bool = True,
    remove_slash: bool = False,
    merge_slashes: bool = True,
    redirect_class: Type[_HTTPMove] = HTTPPermanentRedirect,
) -> _Middleware:
    """Factory for producing a middleware that normalizes the path of a request.

    Normalizing means:
        - Add or remove a trailing slash to the path.
        - Double slashes are replaced by one.

    The middleware returns as soon as it finds a path that resolves
    correctly. The order if both merge and append/remove are enabled is
        1) merge slashes
        2) append/remove slash
        3) both merge slashes and append/remove slash.
    If the path resolves with at least one of those conditions, it will
    redirect to the new path.

    Only one of `append_slash` and `remove_slash` can be enabled. If both
    are `True` the factory will raise an assertion error

    If `append_slash` is `True` the middleware will append a slash when
    needed. If a resource is defined with trailing slash and the request
    comes without it, it will append it automatically.

    If `remove_slash` is `True`, `append_slash` must be `False`. When enabled
    the middleware will remove trailing slashes and redirect if the resource
    is defined

    If merge_slashes is True, merge multiple consecutive slashes in the
    path into one.
    """
    correct_configuration = not (append_slash and remove_slash)
    assert correct_configuration, "Cannot both remove and append slash"

    @middleware
    async def impl(request: Request, handler: Handler) -> StreamResponse:
        if isinstance(request.match_info.route, SystemRoute):
            paths_to_check = []
            if "?" in request.raw_path:
                path, query = request.raw_path.split("?", 1)
                query = "?" + query
            else:
                query = ""
                path = request.raw_path

            if merge_slashes:
                paths_to_check.append(re.sub("//+", "/", path))
            if append_slash and not request.path.endswith("/"):
                paths_to_check.append(path + "/")
            if remove_slash and request.path.endswith("/"):
                paths_to_check.append(path[:-1])
            if merge_slashes and append_slash:
                paths_to_check.append(re.sub("//+", "/", path + "/"))
            if merge_slashes and remove_slash:
                merged_slashes = re.sub("//+", "/", path)
                paths_to_check.append(merged_slashes[:-1])

            for path in paths_to_check:
                path = re.sub("^//+", "/", path)  # SECURITY: GHSA-v6wp-4m6f-gcjg
                resolves, request = await _check_request_resolves(request, path)
                if resolves:
                    raise redirect_class(request.raw_path + query)

        return await handler(request)

    return impl


def _fix_request_current_app(app: "Application") -> _Middleware:
    @middleware
    async def impl(request: Request, handler: Handler) -> StreamResponse:
        with request.match_info.set_current_app(app):
            return await handler(request)

    return impl
