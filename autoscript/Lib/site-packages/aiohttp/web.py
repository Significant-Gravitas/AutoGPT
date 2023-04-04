import asyncio
import logging
import socket
import sys
from argparse import ArgumentParser
from collections.abc import Iterable
from importlib import import_module
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable as TypingIterable,
    List,
    Optional,
    Set,
    Type,
    Union,
    cast,
)

from .abc import AbstractAccessLogger
from .helpers import all_tasks
from .log import access_logger
from .web_app import Application as Application, CleanupError as CleanupError
from .web_exceptions import (
    HTTPAccepted as HTTPAccepted,
    HTTPBadGateway as HTTPBadGateway,
    HTTPBadRequest as HTTPBadRequest,
    HTTPClientError as HTTPClientError,
    HTTPConflict as HTTPConflict,
    HTTPCreated as HTTPCreated,
    HTTPError as HTTPError,
    HTTPException as HTTPException,
    HTTPExpectationFailed as HTTPExpectationFailed,
    HTTPFailedDependency as HTTPFailedDependency,
    HTTPForbidden as HTTPForbidden,
    HTTPFound as HTTPFound,
    HTTPGatewayTimeout as HTTPGatewayTimeout,
    HTTPGone as HTTPGone,
    HTTPInsufficientStorage as HTTPInsufficientStorage,
    HTTPInternalServerError as HTTPInternalServerError,
    HTTPLengthRequired as HTTPLengthRequired,
    HTTPMethodNotAllowed as HTTPMethodNotAllowed,
    HTTPMisdirectedRequest as HTTPMisdirectedRequest,
    HTTPMovedPermanently as HTTPMovedPermanently,
    HTTPMultipleChoices as HTTPMultipleChoices,
    HTTPNetworkAuthenticationRequired as HTTPNetworkAuthenticationRequired,
    HTTPNoContent as HTTPNoContent,
    HTTPNonAuthoritativeInformation as HTTPNonAuthoritativeInformation,
    HTTPNotAcceptable as HTTPNotAcceptable,
    HTTPNotExtended as HTTPNotExtended,
    HTTPNotFound as HTTPNotFound,
    HTTPNotImplemented as HTTPNotImplemented,
    HTTPNotModified as HTTPNotModified,
    HTTPOk as HTTPOk,
    HTTPPartialContent as HTTPPartialContent,
    HTTPPaymentRequired as HTTPPaymentRequired,
    HTTPPermanentRedirect as HTTPPermanentRedirect,
    HTTPPreconditionFailed as HTTPPreconditionFailed,
    HTTPPreconditionRequired as HTTPPreconditionRequired,
    HTTPProxyAuthenticationRequired as HTTPProxyAuthenticationRequired,
    HTTPRedirection as HTTPRedirection,
    HTTPRequestEntityTooLarge as HTTPRequestEntityTooLarge,
    HTTPRequestHeaderFieldsTooLarge as HTTPRequestHeaderFieldsTooLarge,
    HTTPRequestRangeNotSatisfiable as HTTPRequestRangeNotSatisfiable,
    HTTPRequestTimeout as HTTPRequestTimeout,
    HTTPRequestURITooLong as HTTPRequestURITooLong,
    HTTPResetContent as HTTPResetContent,
    HTTPSeeOther as HTTPSeeOther,
    HTTPServerError as HTTPServerError,
    HTTPServiceUnavailable as HTTPServiceUnavailable,
    HTTPSuccessful as HTTPSuccessful,
    HTTPTemporaryRedirect as HTTPTemporaryRedirect,
    HTTPTooManyRequests as HTTPTooManyRequests,
    HTTPUnauthorized as HTTPUnauthorized,
    HTTPUnavailableForLegalReasons as HTTPUnavailableForLegalReasons,
    HTTPUnprocessableEntity as HTTPUnprocessableEntity,
    HTTPUnsupportedMediaType as HTTPUnsupportedMediaType,
    HTTPUpgradeRequired as HTTPUpgradeRequired,
    HTTPUseProxy as HTTPUseProxy,
    HTTPVariantAlsoNegotiates as HTTPVariantAlsoNegotiates,
    HTTPVersionNotSupported as HTTPVersionNotSupported,
)
from .web_fileresponse import FileResponse as FileResponse
from .web_log import AccessLogger
from .web_middlewares import (
    middleware as middleware,
    normalize_path_middleware as normalize_path_middleware,
)
from .web_protocol import (
    PayloadAccessError as PayloadAccessError,
    RequestHandler as RequestHandler,
    RequestPayloadError as RequestPayloadError,
)
from .web_request import (
    BaseRequest as BaseRequest,
    FileField as FileField,
    Request as Request,
)
from .web_response import (
    ContentCoding as ContentCoding,
    Response as Response,
    StreamResponse as StreamResponse,
    json_response as json_response,
)
from .web_routedef import (
    AbstractRouteDef as AbstractRouteDef,
    RouteDef as RouteDef,
    RouteTableDef as RouteTableDef,
    StaticDef as StaticDef,
    delete as delete,
    get as get,
    head as head,
    options as options,
    patch as patch,
    post as post,
    put as put,
    route as route,
    static as static,
    view as view,
)
from .web_runner import (
    AppRunner as AppRunner,
    BaseRunner as BaseRunner,
    BaseSite as BaseSite,
    GracefulExit as GracefulExit,
    NamedPipeSite as NamedPipeSite,
    ServerRunner as ServerRunner,
    SockSite as SockSite,
    TCPSite as TCPSite,
    UnixSite as UnixSite,
)
from .web_server import Server as Server
from .web_urldispatcher import (
    AbstractResource as AbstractResource,
    AbstractRoute as AbstractRoute,
    DynamicResource as DynamicResource,
    PlainResource as PlainResource,
    PrefixedSubAppResource as PrefixedSubAppResource,
    Resource as Resource,
    ResourceRoute as ResourceRoute,
    StaticResource as StaticResource,
    UrlDispatcher as UrlDispatcher,
    UrlMappingMatchInfo as UrlMappingMatchInfo,
    View as View,
)
from .web_ws import (
    WebSocketReady as WebSocketReady,
    WebSocketResponse as WebSocketResponse,
    WSMsgType as WSMsgType,
)

__all__ = (
    # web_app
    "Application",
    "CleanupError",
    # web_exceptions
    "HTTPAccepted",
    "HTTPBadGateway",
    "HTTPBadRequest",
    "HTTPClientError",
    "HTTPConflict",
    "HTTPCreated",
    "HTTPError",
    "HTTPException",
    "HTTPExpectationFailed",
    "HTTPFailedDependency",
    "HTTPForbidden",
    "HTTPFound",
    "HTTPGatewayTimeout",
    "HTTPGone",
    "HTTPInsufficientStorage",
    "HTTPInternalServerError",
    "HTTPLengthRequired",
    "HTTPMethodNotAllowed",
    "HTTPMisdirectedRequest",
    "HTTPMovedPermanently",
    "HTTPMultipleChoices",
    "HTTPNetworkAuthenticationRequired",
    "HTTPNoContent",
    "HTTPNonAuthoritativeInformation",
    "HTTPNotAcceptable",
    "HTTPNotExtended",
    "HTTPNotFound",
    "HTTPNotImplemented",
    "HTTPNotModified",
    "HTTPOk",
    "HTTPPartialContent",
    "HTTPPaymentRequired",
    "HTTPPermanentRedirect",
    "HTTPPreconditionFailed",
    "HTTPPreconditionRequired",
    "HTTPProxyAuthenticationRequired",
    "HTTPRedirection",
    "HTTPRequestEntityTooLarge",
    "HTTPRequestHeaderFieldsTooLarge",
    "HTTPRequestRangeNotSatisfiable",
    "HTTPRequestTimeout",
    "HTTPRequestURITooLong",
    "HTTPResetContent",
    "HTTPSeeOther",
    "HTTPServerError",
    "HTTPServiceUnavailable",
    "HTTPSuccessful",
    "HTTPTemporaryRedirect",
    "HTTPTooManyRequests",
    "HTTPUnauthorized",
    "HTTPUnavailableForLegalReasons",
    "HTTPUnprocessableEntity",
    "HTTPUnsupportedMediaType",
    "HTTPUpgradeRequired",
    "HTTPUseProxy",
    "HTTPVariantAlsoNegotiates",
    "HTTPVersionNotSupported",
    # web_fileresponse
    "FileResponse",
    # web_middlewares
    "middleware",
    "normalize_path_middleware",
    # web_protocol
    "PayloadAccessError",
    "RequestHandler",
    "RequestPayloadError",
    # web_request
    "BaseRequest",
    "FileField",
    "Request",
    # web_response
    "ContentCoding",
    "Response",
    "StreamResponse",
    "json_response",
    # web_routedef
    "AbstractRouteDef",
    "RouteDef",
    "RouteTableDef",
    "StaticDef",
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "route",
    "static",
    "view",
    # web_runner
    "AppRunner",
    "BaseRunner",
    "BaseSite",
    "GracefulExit",
    "ServerRunner",
    "SockSite",
    "TCPSite",
    "UnixSite",
    "NamedPipeSite",
    # web_server
    "Server",
    # web_urldispatcher
    "AbstractResource",
    "AbstractRoute",
    "DynamicResource",
    "PlainResource",
    "PrefixedSubAppResource",
    "Resource",
    "ResourceRoute",
    "StaticResource",
    "UrlDispatcher",
    "UrlMappingMatchInfo",
    "View",
    # web_ws
    "WebSocketReady",
    "WebSocketResponse",
    "WSMsgType",
    # web
    "run_app",
)


try:
    from ssl import SSLContext
except ImportError:  # pragma: no cover
    SSLContext = Any  # type: ignore[misc,assignment]

HostSequence = TypingIterable[str]


async def _run_app(
    app: Union[Application, Awaitable[Application]],
    *,
    host: Optional[Union[str, HostSequence]] = None,
    port: Optional[int] = None,
    path: Optional[str] = None,
    sock: Optional[Union[socket.socket, TypingIterable[socket.socket]]] = None,
    shutdown_timeout: float = 60.0,
    keepalive_timeout: float = 75.0,
    ssl_context: Optional[SSLContext] = None,
    print: Callable[..., None] = print,
    backlog: int = 128,
    access_log_class: Type[AbstractAccessLogger] = AccessLogger,
    access_log_format: str = AccessLogger.LOG_FORMAT,
    access_log: Optional[logging.Logger] = access_logger,
    handle_signals: bool = True,
    reuse_address: Optional[bool] = None,
    reuse_port: Optional[bool] = None,
) -> None:
    # A internal functio to actually do all dirty job for application running
    if asyncio.iscoroutine(app):
        app = await app  # type: ignore[misc]

    app = cast(Application, app)

    runner = AppRunner(
        app,
        handle_signals=handle_signals,
        access_log_class=access_log_class,
        access_log_format=access_log_format,
        access_log=access_log,
        keepalive_timeout=keepalive_timeout,
    )

    await runner.setup()

    sites: List[BaseSite] = []

    try:
        if host is not None:
            if isinstance(host, (str, bytes, bytearray, memoryview)):
                sites.append(
                    TCPSite(
                        runner,
                        host,
                        port,
                        shutdown_timeout=shutdown_timeout,
                        ssl_context=ssl_context,
                        backlog=backlog,
                        reuse_address=reuse_address,
                        reuse_port=reuse_port,
                    )
                )
            else:
                for h in host:
                    sites.append(
                        TCPSite(
                            runner,
                            h,
                            port,
                            shutdown_timeout=shutdown_timeout,
                            ssl_context=ssl_context,
                            backlog=backlog,
                            reuse_address=reuse_address,
                            reuse_port=reuse_port,
                        )
                    )
        elif path is None and sock is None or port is not None:
            sites.append(
                TCPSite(
                    runner,
                    port=port,
                    shutdown_timeout=shutdown_timeout,
                    ssl_context=ssl_context,
                    backlog=backlog,
                    reuse_address=reuse_address,
                    reuse_port=reuse_port,
                )
            )

        if path is not None:
            if isinstance(path, (str, bytes, bytearray, memoryview)):
                sites.append(
                    UnixSite(
                        runner,
                        path,
                        shutdown_timeout=shutdown_timeout,
                        ssl_context=ssl_context,
                        backlog=backlog,
                    )
                )
            else:
                for p in path:
                    sites.append(
                        UnixSite(
                            runner,
                            p,
                            shutdown_timeout=shutdown_timeout,
                            ssl_context=ssl_context,
                            backlog=backlog,
                        )
                    )

        if sock is not None:
            if not isinstance(sock, Iterable):
                sites.append(
                    SockSite(
                        runner,
                        sock,
                        shutdown_timeout=shutdown_timeout,
                        ssl_context=ssl_context,
                        backlog=backlog,
                    )
                )
            else:
                for s in sock:
                    sites.append(
                        SockSite(
                            runner,
                            s,
                            shutdown_timeout=shutdown_timeout,
                            ssl_context=ssl_context,
                            backlog=backlog,
                        )
                    )
        for site in sites:
            await site.start()

        if print:  # pragma: no branch
            names = sorted(str(s.name) for s in runner.sites)
            print(
                "======== Running on {} ========\n"
                "(Press CTRL+C to quit)".format(", ".join(names))
            )

        # sleep forever by 1 hour intervals,
        # on Windows before Python 3.8 wake up every 1 second to handle
        # Ctrl+C smoothly
        if sys.platform == "win32" and sys.version_info < (3, 8):
            delay = 1
        else:
            delay = 3600

        while True:
            await asyncio.sleep(delay)
    finally:
        await runner.cleanup()


def _cancel_tasks(
    to_cancel: Set["asyncio.Task[Any]"], loop: asyncio.AbstractEventLoop
) -> None:
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def run_app(
    app: Union[Application, Awaitable[Application]],
    *,
    host: Optional[Union[str, HostSequence]] = None,
    port: Optional[int] = None,
    path: Optional[str] = None,
    sock: Optional[Union[socket.socket, TypingIterable[socket.socket]]] = None,
    shutdown_timeout: float = 60.0,
    keepalive_timeout: float = 75.0,
    ssl_context: Optional[SSLContext] = None,
    print: Callable[..., None] = print,
    backlog: int = 128,
    access_log_class: Type[AbstractAccessLogger] = AccessLogger,
    access_log_format: str = AccessLogger.LOG_FORMAT,
    access_log: Optional[logging.Logger] = access_logger,
    handle_signals: bool = True,
    reuse_address: Optional[bool] = None,
    reuse_port: Optional[bool] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """Run an app locally"""
    if loop is None:
        loop = asyncio.new_event_loop()

    # Configure if and only if in debugging mode and using the default logger
    if loop.get_debug() and access_log and access_log.name == "aiohttp.access":
        if access_log.level == logging.NOTSET:
            access_log.setLevel(logging.DEBUG)
        if not access_log.hasHandlers():
            access_log.addHandler(logging.StreamHandler())

    main_task = loop.create_task(
        _run_app(
            app,
            host=host,
            port=port,
            path=path,
            sock=sock,
            shutdown_timeout=shutdown_timeout,
            keepalive_timeout=keepalive_timeout,
            ssl_context=ssl_context,
            print=print,
            backlog=backlog,
            access_log_class=access_log_class,
            access_log_format=access_log_format,
            access_log=access_log,
            handle_signals=handle_signals,
            reuse_address=reuse_address,
            reuse_port=reuse_port,
        )
    )

    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main_task)
    except (GracefulExit, KeyboardInterrupt):  # pragma: no cover
        pass
    finally:
        _cancel_tasks({main_task}, loop)
        _cancel_tasks(all_tasks(loop), loop)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def main(argv: List[str]) -> None:
    arg_parser = ArgumentParser(
        description="aiohttp.web Application server", prog="aiohttp.web"
    )
    arg_parser.add_argument(
        "entry_func",
        help=(
            "Callable returning the `aiohttp.web.Application` instance to "
            "run. Should be specified in the 'module:function' syntax."
        ),
        metavar="entry-func",
    )
    arg_parser.add_argument(
        "-H",
        "--hostname",
        help="TCP/IP hostname to serve on (default: %(default)r)",
        default="localhost",
    )
    arg_parser.add_argument(
        "-P",
        "--port",
        help="TCP/IP port to serve on (default: %(default)r)",
        type=int,
        default="8080",
    )
    arg_parser.add_argument(
        "-U",
        "--path",
        help="Unix file system path to serve on. Specifying a path will cause "
        "hostname and port arguments to be ignored.",
    )
    args, extra_argv = arg_parser.parse_known_args(argv)

    # Import logic
    mod_str, _, func_str = args.entry_func.partition(":")
    if not func_str or not mod_str:
        arg_parser.error("'entry-func' not in 'module:function' syntax")
    if mod_str.startswith("."):
        arg_parser.error("relative module names not supported")
    try:
        module = import_module(mod_str)
    except ImportError as ex:
        arg_parser.error(f"unable to import {mod_str}: {ex}")
    try:
        func = getattr(module, func_str)
    except AttributeError:
        arg_parser.error(f"module {mod_str!r} has no attribute {func_str!r}")

    # Compatibility logic
    if args.path is not None and not hasattr(socket, "AF_UNIX"):
        arg_parser.error(
            "file system paths not supported by your operating" " environment"
        )

    logging.basicConfig(level=logging.DEBUG)

    app = func(extra_argv)
    run_app(app, host=args.hostname, port=args.port, path=args.path)
    arg_parser.exit(message="Stopped\n")


if __name__ == "__main__":  # pragma: no branch
    main(sys.argv[1:])  # pragma: no cover
