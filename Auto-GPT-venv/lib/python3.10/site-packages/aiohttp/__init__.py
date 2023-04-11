__version__ = "3.8.4"

from typing import Tuple

from . import hdrs as hdrs
from .client import (
    BaseConnector as BaseConnector,
    ClientConnectionError as ClientConnectionError,
    ClientConnectorCertificateError as ClientConnectorCertificateError,
    ClientConnectorError as ClientConnectorError,
    ClientConnectorSSLError as ClientConnectorSSLError,
    ClientError as ClientError,
    ClientHttpProxyError as ClientHttpProxyError,
    ClientOSError as ClientOSError,
    ClientPayloadError as ClientPayloadError,
    ClientProxyConnectionError as ClientProxyConnectionError,
    ClientRequest as ClientRequest,
    ClientResponse as ClientResponse,
    ClientResponseError as ClientResponseError,
    ClientSession as ClientSession,
    ClientSSLError as ClientSSLError,
    ClientTimeout as ClientTimeout,
    ClientWebSocketResponse as ClientWebSocketResponse,
    ContentTypeError as ContentTypeError,
    Fingerprint as Fingerprint,
    InvalidURL as InvalidURL,
    NamedPipeConnector as NamedPipeConnector,
    RequestInfo as RequestInfo,
    ServerConnectionError as ServerConnectionError,
    ServerDisconnectedError as ServerDisconnectedError,
    ServerFingerprintMismatch as ServerFingerprintMismatch,
    ServerTimeoutError as ServerTimeoutError,
    TCPConnector as TCPConnector,
    TooManyRedirects as TooManyRedirects,
    UnixConnector as UnixConnector,
    WSServerHandshakeError as WSServerHandshakeError,
    request as request,
)
from .cookiejar import CookieJar as CookieJar, DummyCookieJar as DummyCookieJar
from .formdata import FormData as FormData
from .helpers import BasicAuth, ChainMapProxy, ETag
from .http import (
    HttpVersion as HttpVersion,
    HttpVersion10 as HttpVersion10,
    HttpVersion11 as HttpVersion11,
    WebSocketError as WebSocketError,
    WSCloseCode as WSCloseCode,
    WSMessage as WSMessage,
    WSMsgType as WSMsgType,
)
from .multipart import (
    BadContentDispositionHeader as BadContentDispositionHeader,
    BadContentDispositionParam as BadContentDispositionParam,
    BodyPartReader as BodyPartReader,
    MultipartReader as MultipartReader,
    MultipartWriter as MultipartWriter,
    content_disposition_filename as content_disposition_filename,
    parse_content_disposition as parse_content_disposition,
)
from .payload import (
    PAYLOAD_REGISTRY as PAYLOAD_REGISTRY,
    AsyncIterablePayload as AsyncIterablePayload,
    BufferedReaderPayload as BufferedReaderPayload,
    BytesIOPayload as BytesIOPayload,
    BytesPayload as BytesPayload,
    IOBasePayload as IOBasePayload,
    JsonPayload as JsonPayload,
    Payload as Payload,
    StringIOPayload as StringIOPayload,
    StringPayload as StringPayload,
    TextIOPayload as TextIOPayload,
    get_payload as get_payload,
    payload_type as payload_type,
)
from .payload_streamer import streamer as streamer
from .resolver import (
    AsyncResolver as AsyncResolver,
    DefaultResolver as DefaultResolver,
    ThreadedResolver as ThreadedResolver,
)
from .streams import (
    EMPTY_PAYLOAD as EMPTY_PAYLOAD,
    DataQueue as DataQueue,
    EofStream as EofStream,
    FlowControlDataQueue as FlowControlDataQueue,
    StreamReader as StreamReader,
)
from .tracing import (
    TraceConfig as TraceConfig,
    TraceConnectionCreateEndParams as TraceConnectionCreateEndParams,
    TraceConnectionCreateStartParams as TraceConnectionCreateStartParams,
    TraceConnectionQueuedEndParams as TraceConnectionQueuedEndParams,
    TraceConnectionQueuedStartParams as TraceConnectionQueuedStartParams,
    TraceConnectionReuseconnParams as TraceConnectionReuseconnParams,
    TraceDnsCacheHitParams as TraceDnsCacheHitParams,
    TraceDnsCacheMissParams as TraceDnsCacheMissParams,
    TraceDnsResolveHostEndParams as TraceDnsResolveHostEndParams,
    TraceDnsResolveHostStartParams as TraceDnsResolveHostStartParams,
    TraceRequestChunkSentParams as TraceRequestChunkSentParams,
    TraceRequestEndParams as TraceRequestEndParams,
    TraceRequestExceptionParams as TraceRequestExceptionParams,
    TraceRequestRedirectParams as TraceRequestRedirectParams,
    TraceRequestStartParams as TraceRequestStartParams,
    TraceResponseChunkReceivedParams as TraceResponseChunkReceivedParams,
)

__all__: Tuple[str, ...] = (
    "hdrs",
    # client
    "BaseConnector",
    "ClientConnectionError",
    "ClientConnectorCertificateError",
    "ClientConnectorError",
    "ClientConnectorSSLError",
    "ClientError",
    "ClientHttpProxyError",
    "ClientOSError",
    "ClientPayloadError",
    "ClientProxyConnectionError",
    "ClientResponse",
    "ClientRequest",
    "ClientResponseError",
    "ClientSSLError",
    "ClientSession",
    "ClientTimeout",
    "ClientWebSocketResponse",
    "ContentTypeError",
    "Fingerprint",
    "InvalidURL",
    "RequestInfo",
    "ServerConnectionError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ServerTimeoutError",
    "TCPConnector",
    "TooManyRedirects",
    "UnixConnector",
    "NamedPipeConnector",
    "WSServerHandshakeError",
    "request",
    # cookiejar
    "CookieJar",
    "DummyCookieJar",
    # formdata
    "FormData",
    # helpers
    "BasicAuth",
    "ChainMapProxy",
    "ETag",
    # http
    "HttpVersion",
    "HttpVersion10",
    "HttpVersion11",
    "WSMsgType",
    "WSCloseCode",
    "WSMessage",
    "WebSocketError",
    # multipart
    "BadContentDispositionHeader",
    "BadContentDispositionParam",
    "BodyPartReader",
    "MultipartReader",
    "MultipartWriter",
    "content_disposition_filename",
    "parse_content_disposition",
    # payload
    "AsyncIterablePayload",
    "BufferedReaderPayload",
    "BytesIOPayload",
    "BytesPayload",
    "IOBasePayload",
    "JsonPayload",
    "PAYLOAD_REGISTRY",
    "Payload",
    "StringIOPayload",
    "StringPayload",
    "TextIOPayload",
    "get_payload",
    "payload_type",
    # payload_streamer
    "streamer",
    # resolver
    "AsyncResolver",
    "DefaultResolver",
    "ThreadedResolver",
    # streams
    "DataQueue",
    "EMPTY_PAYLOAD",
    "EofStream",
    "FlowControlDataQueue",
    "StreamReader",
    # tracing
    "TraceConfig",
    "TraceConnectionCreateEndParams",
    "TraceConnectionCreateStartParams",
    "TraceConnectionQueuedEndParams",
    "TraceConnectionQueuedStartParams",
    "TraceConnectionReuseconnParams",
    "TraceDnsCacheHitParams",
    "TraceDnsCacheMissParams",
    "TraceDnsResolveHostEndParams",
    "TraceDnsResolveHostStartParams",
    "TraceRequestChunkSentParams",
    "TraceRequestEndParams",
    "TraceRequestExceptionParams",
    "TraceRequestRedirectParams",
    "TraceRequestStartParams",
    "TraceResponseChunkReceivedParams",
)

try:
    from .worker import GunicornUVLoopWebWorker, GunicornWebWorker

    __all__ += ("GunicornWebWorker", "GunicornUVLoopWebWorker")
except ImportError:  # pragma: no cover
    pass
