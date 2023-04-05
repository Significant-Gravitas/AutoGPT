import warnings
from typing import Any, Dict, Iterable, List, Optional, Set  # noqa

from yarl import URL

from .typedefs import LooseHeaders, StrOrURL
from .web_response import Response

__all__ = (
    "HTTPException",
    "HTTPError",
    "HTTPRedirection",
    "HTTPSuccessful",
    "HTTPOk",
    "HTTPCreated",
    "HTTPAccepted",
    "HTTPNonAuthoritativeInformation",
    "HTTPNoContent",
    "HTTPResetContent",
    "HTTPPartialContent",
    "HTTPMultipleChoices",
    "HTTPMovedPermanently",
    "HTTPFound",
    "HTTPSeeOther",
    "HTTPNotModified",
    "HTTPUseProxy",
    "HTTPTemporaryRedirect",
    "HTTPPermanentRedirect",
    "HTTPClientError",
    "HTTPBadRequest",
    "HTTPUnauthorized",
    "HTTPPaymentRequired",
    "HTTPForbidden",
    "HTTPNotFound",
    "HTTPMethodNotAllowed",
    "HTTPNotAcceptable",
    "HTTPProxyAuthenticationRequired",
    "HTTPRequestTimeout",
    "HTTPConflict",
    "HTTPGone",
    "HTTPLengthRequired",
    "HTTPPreconditionFailed",
    "HTTPRequestEntityTooLarge",
    "HTTPRequestURITooLong",
    "HTTPUnsupportedMediaType",
    "HTTPRequestRangeNotSatisfiable",
    "HTTPExpectationFailed",
    "HTTPMisdirectedRequest",
    "HTTPUnprocessableEntity",
    "HTTPFailedDependency",
    "HTTPUpgradeRequired",
    "HTTPPreconditionRequired",
    "HTTPTooManyRequests",
    "HTTPRequestHeaderFieldsTooLarge",
    "HTTPUnavailableForLegalReasons",
    "HTTPServerError",
    "HTTPInternalServerError",
    "HTTPNotImplemented",
    "HTTPBadGateway",
    "HTTPServiceUnavailable",
    "HTTPGatewayTimeout",
    "HTTPVersionNotSupported",
    "HTTPVariantAlsoNegotiates",
    "HTTPInsufficientStorage",
    "HTTPNotExtended",
    "HTTPNetworkAuthenticationRequired",
)


############################################################
# HTTP Exceptions
############################################################


class HTTPException(Response, Exception):

    # You should set in subclasses:
    # status = 200

    status_code = -1
    empty_body = False

    __http_exception__ = True

    def __init__(
        self,
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        body: Any = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> None:
        if body is not None:
            warnings.warn(
                "body argument is deprecated for http web exceptions",
                DeprecationWarning,
            )
        Response.__init__(
            self,
            status=self.status_code,
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        Exception.__init__(self, self.reason)
        if self.body is None and not self.empty_body:
            self.text = f"{self.status}: {self.reason}"

    def __bool__(self) -> bool:
        return True


class HTTPError(HTTPException):
    """Base class for exceptions with status codes in the 400s and 500s."""


class HTTPRedirection(HTTPException):
    """Base class for exceptions with status codes in the 300s."""


class HTTPSuccessful(HTTPException):
    """Base class for exceptions with status codes in the 200s."""


class HTTPOk(HTTPSuccessful):
    status_code = 200


class HTTPCreated(HTTPSuccessful):
    status_code = 201


class HTTPAccepted(HTTPSuccessful):
    status_code = 202


class HTTPNonAuthoritativeInformation(HTTPSuccessful):
    status_code = 203


class HTTPNoContent(HTTPSuccessful):
    status_code = 204
    empty_body = True


class HTTPResetContent(HTTPSuccessful):
    status_code = 205
    empty_body = True


class HTTPPartialContent(HTTPSuccessful):
    status_code = 206


############################################################
# 3xx redirection
############################################################


class _HTTPMove(HTTPRedirection):
    def __init__(
        self,
        location: StrOrURL,
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        body: Any = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> None:
        if not location:
            raise ValueError("HTTP redirects need a location to redirect to.")
        super().__init__(
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        self.headers["Location"] = str(URL(location))
        self.location = location


class HTTPMultipleChoices(_HTTPMove):
    status_code = 300


class HTTPMovedPermanently(_HTTPMove):
    status_code = 301


class HTTPFound(_HTTPMove):
    status_code = 302


# This one is safe after a POST (the redirected location will be
# retrieved with GET):
class HTTPSeeOther(_HTTPMove):
    status_code = 303


class HTTPNotModified(HTTPRedirection):
    # FIXME: this should include a date or etag header
    status_code = 304
    empty_body = True


class HTTPUseProxy(_HTTPMove):
    # Not a move, but looks a little like one
    status_code = 305


class HTTPTemporaryRedirect(_HTTPMove):
    status_code = 307


class HTTPPermanentRedirect(_HTTPMove):
    status_code = 308


############################################################
# 4xx client error
############################################################


class HTTPClientError(HTTPError):
    pass


class HTTPBadRequest(HTTPClientError):
    status_code = 400


class HTTPUnauthorized(HTTPClientError):
    status_code = 401


class HTTPPaymentRequired(HTTPClientError):
    status_code = 402


class HTTPForbidden(HTTPClientError):
    status_code = 403


class HTTPNotFound(HTTPClientError):
    status_code = 404


class HTTPMethodNotAllowed(HTTPClientError):
    status_code = 405

    def __init__(
        self,
        method: str,
        allowed_methods: Iterable[str],
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        body: Any = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> None:
        allow = ",".join(sorted(allowed_methods))
        super().__init__(
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        self.headers["Allow"] = allow
        self.allowed_methods: Set[str] = set(allowed_methods)
        self.method = method.upper()


class HTTPNotAcceptable(HTTPClientError):
    status_code = 406


class HTTPProxyAuthenticationRequired(HTTPClientError):
    status_code = 407


class HTTPRequestTimeout(HTTPClientError):
    status_code = 408


class HTTPConflict(HTTPClientError):
    status_code = 409


class HTTPGone(HTTPClientError):
    status_code = 410


class HTTPLengthRequired(HTTPClientError):
    status_code = 411


class HTTPPreconditionFailed(HTTPClientError):
    status_code = 412


class HTTPRequestEntityTooLarge(HTTPClientError):
    status_code = 413

    def __init__(self, max_size: float, actual_size: float, **kwargs: Any) -> None:
        kwargs.setdefault(
            "text",
            "Maximum request body size {} exceeded, "
            "actual body size {}".format(max_size, actual_size),
        )
        super().__init__(**kwargs)


class HTTPRequestURITooLong(HTTPClientError):
    status_code = 414


class HTTPUnsupportedMediaType(HTTPClientError):
    status_code = 415


class HTTPRequestRangeNotSatisfiable(HTTPClientError):
    status_code = 416


class HTTPExpectationFailed(HTTPClientError):
    status_code = 417


class HTTPMisdirectedRequest(HTTPClientError):
    status_code = 421


class HTTPUnprocessableEntity(HTTPClientError):
    status_code = 422


class HTTPFailedDependency(HTTPClientError):
    status_code = 424


class HTTPUpgradeRequired(HTTPClientError):
    status_code = 426


class HTTPPreconditionRequired(HTTPClientError):
    status_code = 428


class HTTPTooManyRequests(HTTPClientError):
    status_code = 429


class HTTPRequestHeaderFieldsTooLarge(HTTPClientError):
    status_code = 431


class HTTPUnavailableForLegalReasons(HTTPClientError):
    status_code = 451

    def __init__(
        self,
        link: str,
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        body: Any = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            headers=headers,
            reason=reason,
            body=body,
            text=text,
            content_type=content_type,
        )
        self.headers["Link"] = '<%s>; rel="blocked-by"' % link
        self.link = link


############################################################
# 5xx Server Error
############################################################
#  Response status codes beginning with the digit "5" indicate cases in
#  which the server is aware that it has erred or is incapable of
#  performing the request. Except when responding to a HEAD request, the
#  server SHOULD include an entity containing an explanation of the error
#  situation, and whether it is a temporary or permanent condition. User
#  agents SHOULD display any included entity to the user. These response
#  codes are applicable to any request method.


class HTTPServerError(HTTPError):
    pass


class HTTPInternalServerError(HTTPServerError):
    status_code = 500


class HTTPNotImplemented(HTTPServerError):
    status_code = 501


class HTTPBadGateway(HTTPServerError):
    status_code = 502


class HTTPServiceUnavailable(HTTPServerError):
    status_code = 503


class HTTPGatewayTimeout(HTTPServerError):
    status_code = 504


class HTTPVersionNotSupported(HTTPServerError):
    status_code = 505


class HTTPVariantAlsoNegotiates(HTTPServerError):
    status_code = 506


class HTTPInsufficientStorage(HTTPServerError):
    status_code = 507


class HTTPNotExtended(HTTPServerError):
    status_code = 510


class HTTPNetworkAuthenticationRequired(HTTPServerError):
    status_code = 511
