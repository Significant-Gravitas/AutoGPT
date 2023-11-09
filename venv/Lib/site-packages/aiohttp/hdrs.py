"""HTTP Headers constants."""

# After changing the file content call ./tools/gen.py
# to regenerate the headers parser
import sys
from typing import Set

from multidict import istr

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

METH_ANY: Final[str] = "*"
METH_CONNECT: Final[str] = "CONNECT"
METH_HEAD: Final[str] = "HEAD"
METH_GET: Final[str] = "GET"
METH_DELETE: Final[str] = "DELETE"
METH_OPTIONS: Final[str] = "OPTIONS"
METH_PATCH: Final[str] = "PATCH"
METH_POST: Final[str] = "POST"
METH_PUT: Final[str] = "PUT"
METH_TRACE: Final[str] = "TRACE"

METH_ALL: Final[Set[str]] = {
    METH_CONNECT,
    METH_HEAD,
    METH_GET,
    METH_DELETE,
    METH_OPTIONS,
    METH_PATCH,
    METH_POST,
    METH_PUT,
    METH_TRACE,
}

ACCEPT: Final[istr] = istr("Accept")
ACCEPT_CHARSET: Final[istr] = istr("Accept-Charset")
ACCEPT_ENCODING: Final[istr] = istr("Accept-Encoding")
ACCEPT_LANGUAGE: Final[istr] = istr("Accept-Language")
ACCEPT_RANGES: Final[istr] = istr("Accept-Ranges")
ACCESS_CONTROL_MAX_AGE: Final[istr] = istr("Access-Control-Max-Age")
ACCESS_CONTROL_ALLOW_CREDENTIALS: Final[istr] = istr("Access-Control-Allow-Credentials")
ACCESS_CONTROL_ALLOW_HEADERS: Final[istr] = istr("Access-Control-Allow-Headers")
ACCESS_CONTROL_ALLOW_METHODS: Final[istr] = istr("Access-Control-Allow-Methods")
ACCESS_CONTROL_ALLOW_ORIGIN: Final[istr] = istr("Access-Control-Allow-Origin")
ACCESS_CONTROL_EXPOSE_HEADERS: Final[istr] = istr("Access-Control-Expose-Headers")
ACCESS_CONTROL_REQUEST_HEADERS: Final[istr] = istr("Access-Control-Request-Headers")
ACCESS_CONTROL_REQUEST_METHOD: Final[istr] = istr("Access-Control-Request-Method")
AGE: Final[istr] = istr("Age")
ALLOW: Final[istr] = istr("Allow")
AUTHORIZATION: Final[istr] = istr("Authorization")
CACHE_CONTROL: Final[istr] = istr("Cache-Control")
CONNECTION: Final[istr] = istr("Connection")
CONTENT_DISPOSITION: Final[istr] = istr("Content-Disposition")
CONTENT_ENCODING: Final[istr] = istr("Content-Encoding")
CONTENT_LANGUAGE: Final[istr] = istr("Content-Language")
CONTENT_LENGTH: Final[istr] = istr("Content-Length")
CONTENT_LOCATION: Final[istr] = istr("Content-Location")
CONTENT_MD5: Final[istr] = istr("Content-MD5")
CONTENT_RANGE: Final[istr] = istr("Content-Range")
CONTENT_TRANSFER_ENCODING: Final[istr] = istr("Content-Transfer-Encoding")
CONTENT_TYPE: Final[istr] = istr("Content-Type")
COOKIE: Final[istr] = istr("Cookie")
DATE: Final[istr] = istr("Date")
DESTINATION: Final[istr] = istr("Destination")
DIGEST: Final[istr] = istr("Digest")
ETAG: Final[istr] = istr("Etag")
EXPECT: Final[istr] = istr("Expect")
EXPIRES: Final[istr] = istr("Expires")
FORWARDED: Final[istr] = istr("Forwarded")
FROM: Final[istr] = istr("From")
HOST: Final[istr] = istr("Host")
IF_MATCH: Final[istr] = istr("If-Match")
IF_MODIFIED_SINCE: Final[istr] = istr("If-Modified-Since")
IF_NONE_MATCH: Final[istr] = istr("If-None-Match")
IF_RANGE: Final[istr] = istr("If-Range")
IF_UNMODIFIED_SINCE: Final[istr] = istr("If-Unmodified-Since")
KEEP_ALIVE: Final[istr] = istr("Keep-Alive")
LAST_EVENT_ID: Final[istr] = istr("Last-Event-ID")
LAST_MODIFIED: Final[istr] = istr("Last-Modified")
LINK: Final[istr] = istr("Link")
LOCATION: Final[istr] = istr("Location")
MAX_FORWARDS: Final[istr] = istr("Max-Forwards")
ORIGIN: Final[istr] = istr("Origin")
PRAGMA: Final[istr] = istr("Pragma")
PROXY_AUTHENTICATE: Final[istr] = istr("Proxy-Authenticate")
PROXY_AUTHORIZATION: Final[istr] = istr("Proxy-Authorization")
RANGE: Final[istr] = istr("Range")
REFERER: Final[istr] = istr("Referer")
RETRY_AFTER: Final[istr] = istr("Retry-After")
SEC_WEBSOCKET_ACCEPT: Final[istr] = istr("Sec-WebSocket-Accept")
SEC_WEBSOCKET_VERSION: Final[istr] = istr("Sec-WebSocket-Version")
SEC_WEBSOCKET_PROTOCOL: Final[istr] = istr("Sec-WebSocket-Protocol")
SEC_WEBSOCKET_EXTENSIONS: Final[istr] = istr("Sec-WebSocket-Extensions")
SEC_WEBSOCKET_KEY: Final[istr] = istr("Sec-WebSocket-Key")
SEC_WEBSOCKET_KEY1: Final[istr] = istr("Sec-WebSocket-Key1")
SERVER: Final[istr] = istr("Server")
SET_COOKIE: Final[istr] = istr("Set-Cookie")
TE: Final[istr] = istr("TE")
TRAILER: Final[istr] = istr("Trailer")
TRANSFER_ENCODING: Final[istr] = istr("Transfer-Encoding")
UPGRADE: Final[istr] = istr("Upgrade")
URI: Final[istr] = istr("URI")
USER_AGENT: Final[istr] = istr("User-Agent")
VARY: Final[istr] = istr("Vary")
VIA: Final[istr] = istr("Via")
WANT_DIGEST: Final[istr] = istr("Want-Digest")
WARNING: Final[istr] = istr("Warning")
WWW_AUTHENTICATE: Final[istr] = istr("WWW-Authenticate")
X_FORWARDED_FOR: Final[istr] = istr("X-Forwarded-For")
X_FORWARDED_HOST: Final[istr] = istr("X-Forwarded-Host")
X_FORWARDED_PROTO: Final[istr] = istr("X-Forwarded-Proto")
