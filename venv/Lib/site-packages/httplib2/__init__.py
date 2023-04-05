# -*- coding: utf-8 -*-
"""Small, fast HTTP client library for Python."""

__author__ = "Joe Gregorio (joe@bitworking.org)"
__copyright__ = "Copyright 2006, Joe Gregorio"
__contributors__ = [
    "Thomas Broyer (t.broyer@ltgt.net)",
    "James Antill",
    "Xavier Verges Farrero",
    "Jonathan Feinberg",
    "Blair Zajac",
    "Sam Ruby",
    "Louis Nyffenegger",
    "Mark Pilgrim",
    "Alex Yu",
    "Lai Han",
]
__license__ = "MIT"
__version__ = "0.22.0"

import base64
import calendar
import copy
import email
import email.feedparser
from email import header
import email.message
import email.utils
import errno
from gettext import gettext as _
import gzip
from hashlib import md5 as _md5
from hashlib import sha1 as _sha
import hmac
import http.client
import io
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.parse
import zlib

try:
    import socks
except ImportError:
    # TODO: remove this fallback and copypasted socksipy module upon py2/3 merge,
    # idea is to have soft-dependency on any compatible module called socks
    from . import socks
from . import auth
from .error import *
from .iri2uri import iri2uri


def has_timeout(timeout):
    if hasattr(socket, "_GLOBAL_DEFAULT_TIMEOUT"):
        return timeout is not None and timeout is not socket._GLOBAL_DEFAULT_TIMEOUT
    return timeout is not None


__all__ = [
    "debuglevel",
    "FailedToDecompressContent",
    "Http",
    "HttpLib2Error",
    "ProxyInfo",
    "RedirectLimit",
    "RedirectMissingLocation",
    "Response",
    "RETRIES",
    "UnimplementedDigestAuthOptionError",
    "UnimplementedHmacDigestAuthOptionError",
]

# The httplib debug level, set to a non-zero value to get debug output
debuglevel = 0

# A request will be tried 'RETRIES' times if it fails at the socket/connection level.
RETRIES = 2


# Open Items:
# -----------

# Are we removing the cached content too soon on PUT (only delete on 200 Maybe?)

# Pluggable cache storage (supports storing the cache in
#   flat files by default. We need a plug-in architecture
#   that can support Berkeley DB and Squid)

# == Known Issues ==
# Does not handle a resource that uses conneg and Last-Modified but no ETag as a cache validator.
# Does not handle Cache-Control: max-stale
# Does not use Age: headers when calculating cache freshness.

# The number of redirections to follow before giving up.
# Note that only GET redirects are automatically followed.
# Will also honor 301 requests by saving that info and never
# requesting that URI again.
DEFAULT_MAX_REDIRECTS = 5

# Which headers are hop-by-hop headers by default
HOP_BY_HOP = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
]

# https://tools.ietf.org/html/rfc7231#section-8.1.3
SAFE_METHODS = ("GET", "HEAD", "OPTIONS", "TRACE")

# To change, assign to `Http().redirect_codes`
REDIRECT_CODES = frozenset((300, 301, 302, 303, 307, 308))


from httplib2 import certs

CA_CERTS = certs.where()

# PROTOCOL_TLS is python 3.5.3+. PROTOCOL_SSLv23 is deprecated.
# Both PROTOCOL_TLS and PROTOCOL_SSLv23 are equivalent and means:
# > Selects the highest protocol version that both the client and server support.
# > Despite the name, this option can select “TLS” protocols as well as “SSL”.
# source: https://docs.python.org/3.5/library/ssl.html#ssl.PROTOCOL_SSLv23

# PROTOCOL_TLS_CLIENT is python 3.10.0+. PROTOCOL_TLS is deprecated.
# > Auto-negotiate the highest protocol version that both the client and server support, and configure the context client-side connections.
# > The protocol enables CERT_REQUIRED and check_hostname by default.
# source: https://docs.python.org/3.10/library/ssl.html#ssl.PROTOCOL_TLS

DEFAULT_TLS_VERSION = getattr(ssl, "PROTOCOL_TLS_CLIENT", None) or getattr(ssl, "PROTOCOL_TLS", None) or getattr(ssl, "PROTOCOL_SSLv23")


def _build_ssl_context(
    disable_ssl_certificate_validation,
    ca_certs,
    cert_file=None,
    key_file=None,
    maximum_version=None,
    minimum_version=None,
    key_password=None,
):
    if not hasattr(ssl, "SSLContext"):
        raise RuntimeError("httplib2 requires Python 3.2+ for ssl.SSLContext")

    context = ssl.SSLContext(DEFAULT_TLS_VERSION)
    # check_hostname and verify_mode should be set in opposite order during disable
    # https://bugs.python.org/issue31431
    if disable_ssl_certificate_validation and hasattr(context, "check_hostname"):
        context.check_hostname = not disable_ssl_certificate_validation
    context.verify_mode = ssl.CERT_NONE if disable_ssl_certificate_validation else ssl.CERT_REQUIRED

    # SSLContext.maximum_version and SSLContext.minimum_version are python 3.7+.
    # source: https://docs.python.org/3/library/ssl.html#ssl.SSLContext.maximum_version
    if maximum_version is not None:
        if hasattr(context, "maximum_version"):
            if isinstance(maximum_version, str):
                maximum_version = getattr(ssl.TLSVersion, maximum_version)
            context.maximum_version = maximum_version
        else:
            raise RuntimeError("setting tls_maximum_version requires Python 3.7 and OpenSSL 1.1 or newer")
    if minimum_version is not None:
        if hasattr(context, "minimum_version"):
            if isinstance(minimum_version, str):
                minimum_version = getattr(ssl.TLSVersion, minimum_version)
            context.minimum_version = minimum_version
        else:
            raise RuntimeError("setting tls_minimum_version requires Python 3.7 and OpenSSL 1.1 or newer")
    # check_hostname requires python 3.4+
    # we will perform the equivalent in HTTPSConnectionWithTimeout.connect() by calling ssl.match_hostname
    # if check_hostname is not supported.
    if hasattr(context, "check_hostname"):
        context.check_hostname = not disable_ssl_certificate_validation

    context.load_verify_locations(ca_certs)

    if cert_file:
        context.load_cert_chain(cert_file, key_file, key_password)

    return context


def _get_end2end_headers(response):
    hopbyhop = list(HOP_BY_HOP)
    hopbyhop.extend([x.strip() for x in response.get("connection", "").split(",")])
    return [header for header in list(response.keys()) if header not in hopbyhop]


_missing = object()


def _errno_from_exception(e):
    # TODO python 3.11+ cheap try: return e.errno except AttributeError: pass
    errno = getattr(e, "errno", _missing)
    if errno is not _missing:
        return errno

    # socket.error and common wrap in .args
    args = getattr(e, "args", None)
    if args:
        return _errno_from_exception(args[0])

    # pysocks.ProxyError wraps in .socket_err
    # https://github.com/httplib2/httplib2/pull/202
    socket_err = getattr(e, "socket_err", None)
    if socket_err:
        return _errno_from_exception(socket_err)

    return None


URI = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


def parse_uri(uri):
    """Parses a URI using the regex given in Appendix B of RFC 3986.

        (scheme, authority, path, query, fragment) = parse_uri(uri)
    """
    groups = URI.match(uri).groups()
    return (groups[1], groups[3], groups[4], groups[6], groups[8])


def urlnorm(uri):
    (scheme, authority, path, query, fragment) = parse_uri(uri)
    if not scheme or not authority:
        raise RelativeURIError("Only absolute URIs are allowed. uri = %s" % uri)
    authority = authority.lower()
    scheme = scheme.lower()
    if not path:
        path = "/"
    # Could do syntax based normalization of the URI before
    # computing the digest. See Section 6.2.2 of Std 66.
    request_uri = query and "?".join([path, query]) or path
    scheme = scheme.lower()
    defrag_uri = scheme + "://" + authority + request_uri
    return scheme, authority, request_uri, defrag_uri


# Cache filename construction (original borrowed from Venus http://intertwingly.net/code/venus/)
re_url_scheme = re.compile(r"^\w+://")
re_unsafe = re.compile(r"[^\w\-_.()=!]+", re.ASCII)


def safename(filename):
    """Return a filename suitable for the cache.
    Strips dangerous and common characters to create a filename we
    can use to store the cache in.
    """
    if isinstance(filename, bytes):
        filename_bytes = filename
        filename = filename.decode("utf-8")
    else:
        filename_bytes = filename.encode("utf-8")
    filemd5 = _md5(filename_bytes).hexdigest()
    filename = re_url_scheme.sub("", filename)
    filename = re_unsafe.sub("", filename)

    # limit length of filename (vital for Windows)
    # https://github.com/httplib2/httplib2/pull/74
    # C:\Users\    <username>    \AppData\Local\Temp\  <safe_filename>  ,   <md5>
    #   9 chars + max 104 chars  +     20 chars      +       x       +  1  +  32  = max 259 chars
    # Thus max safe filename x = 93 chars. Let it be 90 to make a round sum:
    filename = filename[:90]

    return ",".join((filename, filemd5))


NORMALIZE_SPACE = re.compile(r"(?:\r\n)?[ \t]+")


def _normalize_headers(headers):
    return dict(
        [
            (_convert_byte_str(key).lower(), NORMALIZE_SPACE.sub(_convert_byte_str(value), " ").strip(),)
            for (key, value) in headers.items()
        ]
    )


def _convert_byte_str(s):
    if not isinstance(s, str):
        return str(s, "utf-8")
    return s


def _parse_cache_control(headers):
    retval = {}
    if "cache-control" in headers:
        parts = headers["cache-control"].split(",")
        parts_with_args = [
            tuple([x.strip().lower() for x in part.split("=", 1)]) for part in parts if -1 != part.find("=")
        ]
        parts_wo_args = [(name.strip().lower(), 1) for name in parts if -1 == name.find("=")]
        retval = dict(parts_with_args + parts_wo_args)
    return retval


# Whether to use a strict mode to parse WWW-Authenticate headers
# Might lead to bad results in case of ill-formed header value,
# so disabled by default, falling back to relaxed parsing.
# Set to true to turn on, useful for testing servers.
USE_WWW_AUTH_STRICT_PARSING = 0


def _entry_disposition(response_headers, request_headers):
    """Determine freshness from the Date, Expires and Cache-Control headers.

    We don't handle the following:

    1. Cache-Control: max-stale
    2. Age: headers are not used in the calculations.

    Not that this algorithm is simpler than you might think
    because we are operating as a private (non-shared) cache.
    This lets us ignore 's-maxage'. We can also ignore
    'proxy-invalidate' since we aren't a proxy.
    We will never return a stale document as
    fresh as a design decision, and thus the non-implementation
    of 'max-stale'. This also lets us safely ignore 'must-revalidate'
    since we operate as if every server has sent 'must-revalidate'.
    Since we are private we get to ignore both 'public' and
    'private' parameters. We also ignore 'no-transform' since
    we don't do any transformations.
    The 'no-store' parameter is handled at a higher level.
    So the only Cache-Control parameters we look at are:

    no-cache
    only-if-cached
    max-age
    min-fresh
    """

    retval = "STALE"
    cc = _parse_cache_control(request_headers)
    cc_response = _parse_cache_control(response_headers)

    if "pragma" in request_headers and request_headers["pragma"].lower().find("no-cache") != -1:
        retval = "TRANSPARENT"
        if "cache-control" not in request_headers:
            request_headers["cache-control"] = "no-cache"
    elif "no-cache" in cc:
        retval = "TRANSPARENT"
    elif "no-cache" in cc_response:
        retval = "STALE"
    elif "only-if-cached" in cc:
        retval = "FRESH"
    elif "date" in response_headers:
        date = calendar.timegm(email.utils.parsedate_tz(response_headers["date"]))
        now = time.time()
        current_age = max(0, now - date)
        if "max-age" in cc_response:
            try:
                freshness_lifetime = int(cc_response["max-age"])
            except ValueError:
                freshness_lifetime = 0
        elif "expires" in response_headers:
            expires = email.utils.parsedate_tz(response_headers["expires"])
            if None == expires:
                freshness_lifetime = 0
            else:
                freshness_lifetime = max(0, calendar.timegm(expires) - date)
        else:
            freshness_lifetime = 0
        if "max-age" in cc:
            try:
                freshness_lifetime = int(cc["max-age"])
            except ValueError:
                freshness_lifetime = 0
        if "min-fresh" in cc:
            try:
                min_fresh = int(cc["min-fresh"])
            except ValueError:
                min_fresh = 0
            current_age += min_fresh
        if freshness_lifetime > current_age:
            retval = "FRESH"
    return retval


def _decompressContent(response, new_content):
    content = new_content
    try:
        encoding = response.get("content-encoding", None)
        if encoding in ["gzip", "deflate"]:
            if encoding == "gzip":
                content = gzip.GzipFile(fileobj=io.BytesIO(new_content)).read()
            if encoding == "deflate":
                try:
                    content = zlib.decompress(content, zlib.MAX_WBITS)
                except (IOError, zlib.error):
                    content = zlib.decompress(content, -zlib.MAX_WBITS)
            response["content-length"] = str(len(content))
            # Record the historical presence of the encoding in a way the won't interfere.
            response["-content-encoding"] = response["content-encoding"]
            del response["content-encoding"]
    except (IOError, zlib.error):
        content = ""
        raise FailedToDecompressContent(
            _("Content purported to be compressed with %s but failed to decompress.") % response.get("content-encoding"),
            response,
            content,
        )
    return content


def _bind_write_headers(msg):
    def _write_headers(self):
        # Self refers to the Generator object.
        for h, v in msg.items():
            print("%s:" % h, end=" ", file=self._fp)
            if isinstance(v, header.Header):
                print(v.encode(maxlinelen=self._maxheaderlen), file=self._fp)
            else:
                # email.Header got lots of smarts, so use it.
                headers = header.Header(v, maxlinelen=self._maxheaderlen, charset="utf-8", header_name=h)
                print(headers.encode(), file=self._fp)
        # A blank line always separates headers from body.
        print(file=self._fp)

    return _write_headers


def _updateCache(request_headers, response_headers, content, cache, cachekey):
    if cachekey:
        cc = _parse_cache_control(request_headers)
        cc_response = _parse_cache_control(response_headers)
        if "no-store" in cc or "no-store" in cc_response:
            cache.delete(cachekey)
        else:
            info = email.message.Message()
            for key, value in response_headers.items():
                if key not in ["status", "content-encoding", "transfer-encoding"]:
                    info[key] = value

            # Add annotations to the cache to indicate what headers
            # are variant for this request.
            vary = response_headers.get("vary", None)
            if vary:
                vary_headers = vary.lower().replace(" ", "").split(",")
                for header in vary_headers:
                    key = "-varied-%s" % header
                    try:
                        info[key] = request_headers[header]
                    except KeyError:
                        pass

            status = response_headers.status
            if status == 304:
                status = 200

            status_header = "status: %d\r\n" % status

            try:
                header_str = info.as_string()
            except UnicodeEncodeError:
                setattr(info, "_write_headers", _bind_write_headers(info))
                header_str = info.as_string()

            header_str = re.sub("\r(?!\n)|(?<!\r)\n", "\r\n", header_str)
            text = b"".join([status_header.encode("utf-8"), header_str.encode("utf-8"), content])

            cache.set(cachekey, text)


def _cnonce():
    dig = _md5(
        ("%s:%s" % (time.ctime(), ["0123456789"[random.randrange(0, 9)] for i in range(20)])).encode("utf-8")
    ).hexdigest()
    return dig[:16]


def _wsse_username_token(cnonce, iso_now, password):
    return (
        base64.b64encode(_sha(("%s%s%s" % (cnonce, iso_now, password)).encode("utf-8")).digest()).strip().decode("utf-8")
    )


# For credentials we need two things, first
# a pool of credential to try (not necesarily tied to BAsic, Digest, etc.)
# Then we also need a list of URIs that have already demanded authentication
# That list is tricky since sub-URIs can take the same auth, or the
# auth scheme may change as you descend the tree.
# So we also need each Auth instance to be able to tell us
# how close to the 'top' it is.


class Authentication(object):
    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        (scheme, authority, path, query, fragment) = parse_uri(request_uri)
        self.path = path
        self.host = host
        self.credentials = credentials
        self.http = http

    def depth(self, request_uri):
        (scheme, authority, path, query, fragment) = parse_uri(request_uri)
        return request_uri[len(self.path) :].count("/")

    def inscope(self, host, request_uri):
        # XXX Should we normalize the request_uri?
        (scheme, authority, path, query, fragment) = parse_uri(request_uri)
        return (host == self.host) and path.startswith(self.path)

    def request(self, method, request_uri, headers, content):
        """Modify the request headers to add the appropriate
        Authorization header. Over-rise this in sub-classes."""
        pass

    def response(self, response, content):
        """Gives us a chance to update with new nonces
        or such returned from the last authorized response.
        Over-rise this in sub-classes if necessary.

        Return TRUE is the request is to be retried, for
        example Digest may return stale=true.
        """
        return False

    def __eq__(self, auth):
        return False

    def __ne__(self, auth):
        return True

    def __lt__(self, auth):
        return True

    def __gt__(self, auth):
        return False

    def __le__(self, auth):
        return True

    def __ge__(self, auth):
        return False

    def __bool__(self):
        return True


class BasicAuthentication(Authentication):
    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)

    def request(self, method, request_uri, headers, content):
        """Modify the request headers to add the appropriate
        Authorization header."""
        headers["authorization"] = "Basic " + base64.b64encode(
            ("%s:%s" % self.credentials).encode("utf-8")
        ).strip().decode("utf-8")


class DigestAuthentication(Authentication):
    """Only do qop='auth' and MD5, since that
    is all Apache currently implements"""

    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)
        self.challenge = auth._parse_www_authenticate(response, "www-authenticate")["digest"]
        qop = self.challenge.get("qop", "auth")
        self.challenge["qop"] = ("auth" in [x.strip() for x in qop.split()]) and "auth" or None
        if self.challenge["qop"] is None:
            raise UnimplementedDigestAuthOptionError(_("Unsupported value for qop: %s." % qop))
        self.challenge["algorithm"] = self.challenge.get("algorithm", "MD5").upper()
        if self.challenge["algorithm"] != "MD5":
            raise UnimplementedDigestAuthOptionError(
                _("Unsupported value for algorithm: %s." % self.challenge["algorithm"])
            )
        self.A1 = "".join([self.credentials[0], ":", self.challenge["realm"], ":", self.credentials[1],])
        self.challenge["nc"] = 1

    def request(self, method, request_uri, headers, content, cnonce=None):
        """Modify the request headers"""
        H = lambda x: _md5(x.encode("utf-8")).hexdigest()
        KD = lambda s, d: H("%s:%s" % (s, d))
        A2 = "".join([method, ":", request_uri])
        self.challenge["cnonce"] = cnonce or _cnonce()
        request_digest = '"%s"' % KD(
            H(self.A1),
            "%s:%s:%s:%s:%s"
            % (
                self.challenge["nonce"],
                "%08x" % self.challenge["nc"],
                self.challenge["cnonce"],
                self.challenge["qop"],
                H(A2),
            ),
        )
        headers["authorization"] = (
            'Digest username="%s", realm="%s", nonce="%s", '
            'uri="%s", algorithm=%s, response=%s, qop=%s, '
            'nc=%08x, cnonce="%s"'
        ) % (
            self.credentials[0],
            self.challenge["realm"],
            self.challenge["nonce"],
            request_uri,
            self.challenge["algorithm"],
            request_digest,
            self.challenge["qop"],
            self.challenge["nc"],
            self.challenge["cnonce"],
        )
        if self.challenge.get("opaque"):
            headers["authorization"] += ', opaque="%s"' % self.challenge["opaque"]
        self.challenge["nc"] += 1

    def response(self, response, content):
        if "authentication-info" not in response:
            challenge = auth._parse_www_authenticate(response, "www-authenticate").get("digest", {})
            if "true" == challenge.get("stale"):
                self.challenge["nonce"] = challenge["nonce"]
                self.challenge["nc"] = 1
                return True
        else:
            updated_challenge = auth._parse_authentication_info(response, "authentication-info")

            if "nextnonce" in updated_challenge:
                self.challenge["nonce"] = updated_challenge["nextnonce"]
                self.challenge["nc"] = 1
        return False


class HmacDigestAuthentication(Authentication):
    """Adapted from Robert Sayre's code and DigestAuthentication above."""

    __author__ = "Thomas Broyer (t.broyer@ltgt.net)"

    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)
        challenge = auth._parse_www_authenticate(response, "www-authenticate")
        self.challenge = challenge["hmacdigest"]
        # TODO: self.challenge['domain']
        self.challenge["reason"] = self.challenge.get("reason", "unauthorized")
        if self.challenge["reason"] not in ["unauthorized", "integrity"]:
            self.challenge["reason"] = "unauthorized"
        self.challenge["salt"] = self.challenge.get("salt", "")
        if not self.challenge.get("snonce"):
            raise UnimplementedHmacDigestAuthOptionError(
                _("The challenge doesn't contain a server nonce, or this one is empty.")
            )
        self.challenge["algorithm"] = self.challenge.get("algorithm", "HMAC-SHA-1")
        if self.challenge["algorithm"] not in ["HMAC-SHA-1", "HMAC-MD5"]:
            raise UnimplementedHmacDigestAuthOptionError(
                _("Unsupported value for algorithm: %s." % self.challenge["algorithm"])
            )
        self.challenge["pw-algorithm"] = self.challenge.get("pw-algorithm", "SHA-1")
        if self.challenge["pw-algorithm"] not in ["SHA-1", "MD5"]:
            raise UnimplementedHmacDigestAuthOptionError(
                _("Unsupported value for pw-algorithm: %s." % self.challenge["pw-algorithm"])
            )
        if self.challenge["algorithm"] == "HMAC-MD5":
            self.hashmod = _md5
        else:
            self.hashmod = _sha
        if self.challenge["pw-algorithm"] == "MD5":
            self.pwhashmod = _md5
        else:
            self.pwhashmod = _sha
        self.key = "".join(
            [
                self.credentials[0],
                ":",
                self.pwhashmod.new("".join([self.credentials[1], self.challenge["salt"]])).hexdigest().lower(),
                ":",
                self.challenge["realm"],
            ]
        )
        self.key = self.pwhashmod.new(self.key).hexdigest().lower()

    def request(self, method, request_uri, headers, content):
        """Modify the request headers"""
        keys = _get_end2end_headers(headers)
        keylist = "".join(["%s " % k for k in keys])
        headers_val = "".join([headers[k] for k in keys])
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cnonce = _cnonce()
        request_digest = "%s:%s:%s:%s:%s" % (method, request_uri, cnonce, self.challenge["snonce"], headers_val,)
        request_digest = hmac.new(self.key, request_digest, self.hashmod).hexdigest().lower()
        headers["authorization"] = (
            'HMACDigest username="%s", realm="%s", snonce="%s",'
            ' cnonce="%s", uri="%s", created="%s", '
            'response="%s", headers="%s"'
        ) % (
            self.credentials[0],
            self.challenge["realm"],
            self.challenge["snonce"],
            cnonce,
            request_uri,
            created,
            request_digest,
            keylist,
        )

    def response(self, response, content):
        challenge = auth._parse_www_authenticate(response, "www-authenticate").get("hmacdigest", {})
        if challenge.get("reason") in ["integrity", "stale"]:
            return True
        return False


class WsseAuthentication(Authentication):
    """This is thinly tested and should not be relied upon.
    At this time there isn't any third party server to test against.
    Blogger and TypePad implemented this algorithm at one point
    but Blogger has since switched to Basic over HTTPS and
    TypePad has implemented it wrong, by never issuing a 401
    challenge but instead requiring your client to telepathically know that
    their endpoint is expecting WSSE profile="UsernameToken"."""

    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)

    def request(self, method, request_uri, headers, content):
        """Modify the request headers to add the appropriate
        Authorization header."""
        headers["authorization"] = 'WSSE profile="UsernameToken"'
        iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cnonce = _cnonce()
        password_digest = _wsse_username_token(cnonce, iso_now, self.credentials[1])
        headers["X-WSSE"] = ('UsernameToken Username="%s", PasswordDigest="%s", ' 'Nonce="%s", Created="%s"') % (
            self.credentials[0],
            password_digest,
            cnonce,
            iso_now,
        )


class GoogleLoginAuthentication(Authentication):
    def __init__(self, credentials, host, request_uri, headers, response, content, http):
        from urllib.parse import urlencode

        Authentication.__init__(self, credentials, host, request_uri, headers, response, content, http)
        challenge = auth._parse_www_authenticate(response, "www-authenticate")
        service = challenge["googlelogin"].get("service", "xapi")
        # Bloggger actually returns the service in the challenge
        # For the rest we guess based on the URI
        if service == "xapi" and request_uri.find("calendar") > 0:
            service = "cl"
        # No point in guessing Base or Spreadsheet
        # elif request_uri.find("spreadsheets") > 0:
        #    service = "wise"

        auth = dict(Email=credentials[0], Passwd=credentials[1], service=service, source=headers["user-agent"],)
        resp, content = self.http.request(
            "https://www.google.com/accounts/ClientLogin",
            method="POST",
            body=urlencode(auth),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        lines = content.split("\n")
        d = dict([tuple(line.split("=", 1)) for line in lines if line])
        if resp.status == 403:
            self.Auth = ""
        else:
            self.Auth = d["Auth"]

    def request(self, method, request_uri, headers, content):
        """Modify the request headers to add the appropriate
        Authorization header."""
        headers["authorization"] = "GoogleLogin Auth=" + self.Auth


AUTH_SCHEME_CLASSES = {
    "basic": BasicAuthentication,
    "wsse": WsseAuthentication,
    "digest": DigestAuthentication,
    "hmacdigest": HmacDigestAuthentication,
    "googlelogin": GoogleLoginAuthentication,
}

AUTH_SCHEME_ORDER = ["hmacdigest", "googlelogin", "digest", "wsse", "basic"]


class FileCache(object):
    """Uses a local directory as a store for cached files.
    Not really safe to use if multiple threads or processes are going to
    be running on the same cache.
    """

    def __init__(self, cache, safe=safename):  # use safe=lambda x: md5.new(x).hexdigest() for the old behavior
        self.cache = cache
        self.safe = safe
        if not os.path.exists(cache):
            os.makedirs(self.cache)

    def get(self, key):
        retval = None
        cacheFullPath = os.path.join(self.cache, self.safe(key))
        try:
            f = open(cacheFullPath, "rb")
            retval = f.read()
            f.close()
        except IOError:
            pass
        return retval

    def set(self, key, value):
        cacheFullPath = os.path.join(self.cache, self.safe(key))
        f = open(cacheFullPath, "wb")
        f.write(value)
        f.close()

    def delete(self, key):
        cacheFullPath = os.path.join(self.cache, self.safe(key))
        if os.path.exists(cacheFullPath):
            os.remove(cacheFullPath)


class Credentials(object):
    def __init__(self):
        self.credentials = []

    def add(self, name, password, domain=""):
        self.credentials.append((domain.lower(), name, password))

    def clear(self):
        self.credentials = []

    def iter(self, domain):
        for (cdomain, name, password) in self.credentials:
            if cdomain == "" or domain == cdomain:
                yield (name, password)


class KeyCerts(Credentials):
    """Identical to Credentials except that
    name/password are mapped to key/cert."""

    def add(self, key, cert, domain, password):
        self.credentials.append((domain.lower(), key, cert, password))

    def iter(self, domain):
        for (cdomain, key, cert, password) in self.credentials:
            if cdomain == "" or domain == cdomain:
                yield (key, cert, password)


class AllHosts(object):
    pass


class ProxyInfo(object):
    """Collect information required to use a proxy."""

    bypass_hosts = ()

    def __init__(
        self, proxy_type, proxy_host, proxy_port, proxy_rdns=True, proxy_user=None, proxy_pass=None, proxy_headers=None,
    ):
        """Args:

          proxy_type: The type of proxy server.  This must be set to one of
          socks.PROXY_TYPE_XXX constants.  For example:  p =
          ProxyInfo(proxy_type=socks.PROXY_TYPE_HTTP, proxy_host='localhost',
          proxy_port=8000)
          proxy_host: The hostname or IP address of the proxy server.
          proxy_port: The port that the proxy server is running on.
          proxy_rdns: If True (default), DNS queries will not be performed
          locally, and instead, handed to the proxy to resolve.  This is useful
          if the network does not allow resolution of non-local names. In
          httplib2 0.9 and earlier, this defaulted to False.
          proxy_user: The username used to authenticate with the proxy server.
          proxy_pass: The password used to authenticate with the proxy server.
          proxy_headers: Additional or modified headers for the proxy connect
          request.
        """
        if isinstance(proxy_user, bytes):
            proxy_user = proxy_user.decode()
        if isinstance(proxy_pass, bytes):
            proxy_pass = proxy_pass.decode()
        (
            self.proxy_type,
            self.proxy_host,
            self.proxy_port,
            self.proxy_rdns,
            self.proxy_user,
            self.proxy_pass,
            self.proxy_headers,
        ) = (
            proxy_type,
            proxy_host,
            proxy_port,
            proxy_rdns,
            proxy_user,
            proxy_pass,
            proxy_headers,
        )

    def astuple(self):
        return (
            self.proxy_type,
            self.proxy_host,
            self.proxy_port,
            self.proxy_rdns,
            self.proxy_user,
            self.proxy_pass,
            self.proxy_headers,
        )

    def isgood(self):
        return socks and (self.proxy_host != None) and (self.proxy_port != None)

    def applies_to(self, hostname):
        return not self.bypass_host(hostname)

    def bypass_host(self, hostname):
        """Has this host been excluded from the proxy config"""
        if self.bypass_hosts is AllHosts:
            return True

        hostname = "." + hostname.lstrip(".")
        for skip_name in self.bypass_hosts:
            # *.suffix
            if skip_name.startswith(".") and hostname.endswith(skip_name):
                return True
            # exact match
            if hostname == "." + skip_name:
                return True
        return False

    def __repr__(self):
        return (
            "<ProxyInfo type={p.proxy_type} "
            "host:port={p.proxy_host}:{p.proxy_port} rdns={p.proxy_rdns}"
            + " user={p.proxy_user} headers={p.proxy_headers}>"
        ).format(p=self)


def proxy_info_from_environment(method="http"):
    """Read proxy info from the environment variables.
    """
    if method not in ("http", "https"):
        return

    env_var = method + "_proxy"
    url = os.environ.get(env_var, os.environ.get(env_var.upper()))
    if not url:
        return
    return proxy_info_from_url(url, method, noproxy=None)


def proxy_info_from_url(url, method="http", noproxy=None):
    """Construct a ProxyInfo from a URL (such as http_proxy env var)
    """
    url = urllib.parse.urlparse(url)

    proxy_type = 3  # socks.PROXY_TYPE_HTTP
    pi = ProxyInfo(
        proxy_type=proxy_type,
        proxy_host=url.hostname,
        proxy_port=url.port or dict(https=443, http=80)[method],
        proxy_user=url.username or None,
        proxy_pass=url.password or None,
        proxy_headers=None,
    )

    bypass_hosts = []
    # If not given an explicit noproxy value, respect values in env vars.
    if noproxy is None:
        noproxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
    # Special case: A single '*' character means all hosts should be bypassed.
    if noproxy == "*":
        bypass_hosts = AllHosts
    elif noproxy.strip():
        bypass_hosts = noproxy.split(",")
        bypass_hosts = tuple(filter(bool, bypass_hosts))  # To exclude empty string.

    pi.bypass_hosts = bypass_hosts
    return pi


class HTTPConnectionWithTimeout(http.client.HTTPConnection):
    """HTTPConnection subclass that supports timeouts

    HTTPConnection subclass that supports timeouts

    All timeouts are in seconds. If None is passed for timeout then
    Python's default timeout for sockets will be used. See for example
    the docs of socket.setdefaulttimeout():
    http://docs.python.org/library/socket.html#socket.setdefaulttimeout
    """

    def __init__(self, host, port=None, timeout=None, proxy_info=None):
        http.client.HTTPConnection.__init__(self, host, port=port, timeout=timeout)

        self.proxy_info = proxy_info
        if proxy_info and not isinstance(proxy_info, ProxyInfo):
            self.proxy_info = proxy_info("http")

    def connect(self):
        """Connect to the host and port specified in __init__."""
        if self.proxy_info and socks is None:
            raise ProxiesUnavailableError("Proxy support missing but proxy use was requested!")
        if self.proxy_info and self.proxy_info.isgood() and self.proxy_info.applies_to(self.host):
            use_proxy = True
            (
                proxy_type,
                proxy_host,
                proxy_port,
                proxy_rdns,
                proxy_user,
                proxy_pass,
                proxy_headers,
            ) = self.proxy_info.astuple()

            host = proxy_host
            port = proxy_port
        else:
            use_proxy = False

            host = self.host
            port = self.port
            proxy_type = None

        socket_err = None

        for res in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            try:
                if use_proxy:
                    self.sock = socks.socksocket(af, socktype, proto)
                    self.sock.setproxy(
                        proxy_type, proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass,
                    )
                else:
                    self.sock = socket.socket(af, socktype, proto)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if has_timeout(self.timeout):
                    self.sock.settimeout(self.timeout)
                if self.debuglevel > 0:
                    print("connect: ({0}, {1}) ************".format(self.host, self.port))
                    if use_proxy:
                        print(
                            "proxy: {0} ************".format(
                                str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers,))
                            )
                        )

                self.sock.connect((self.host, self.port) + sa[2:])
            except socket.error as e:
                socket_err = e
                if self.debuglevel > 0:
                    print("connect fail: ({0}, {1})".format(self.host, self.port))
                    if use_proxy:
                        print(
                            "proxy: {0}".format(
                                str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers,))
                            )
                        )
                if self.sock:
                    self.sock.close()
                self.sock = None
                continue
            break
        if not self.sock:
            raise socket_err


class HTTPSConnectionWithTimeout(http.client.HTTPSConnection):
    """This class allows communication via SSL.

    All timeouts are in seconds. If None is passed for timeout then
    Python's default timeout for sockets will be used. See for example
    the docs of socket.setdefaulttimeout():
    http://docs.python.org/library/socket.html#socket.setdefaulttimeout
    """

    def __init__(
        self,
        host,
        port=None,
        key_file=None,
        cert_file=None,
        timeout=None,
        proxy_info=None,
        ca_certs=None,
        disable_ssl_certificate_validation=False,
        tls_maximum_version=None,
        tls_minimum_version=None,
        key_password=None,
    ):

        self.disable_ssl_certificate_validation = disable_ssl_certificate_validation
        self.ca_certs = ca_certs if ca_certs else CA_CERTS

        self.proxy_info = proxy_info
        if proxy_info and not isinstance(proxy_info, ProxyInfo):
            self.proxy_info = proxy_info("https")

        context = _build_ssl_context(
            self.disable_ssl_certificate_validation,
            self.ca_certs,
            cert_file,
            key_file,
            maximum_version=tls_maximum_version,
            minimum_version=tls_minimum_version,
            key_password=key_password,
        )
        super(HTTPSConnectionWithTimeout, self).__init__(
            host, port=port, timeout=timeout, context=context,
        )
        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password

    def connect(self):
        """Connect to a host on a given (SSL) port."""
        if self.proxy_info and self.proxy_info.isgood() and self.proxy_info.applies_to(self.host):
            use_proxy = True
            (
                proxy_type,
                proxy_host,
                proxy_port,
                proxy_rdns,
                proxy_user,
                proxy_pass,
                proxy_headers,
            ) = self.proxy_info.astuple()

            host = proxy_host
            port = proxy_port
        else:
            use_proxy = False

            host = self.host
            port = self.port
            proxy_type = None
            proxy_headers = None

        socket_err = None

        address_info = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        for family, socktype, proto, canonname, sockaddr in address_info:
            try:
                if use_proxy:
                    sock = socks.socksocket(family, socktype, proto)

                    sock.setproxy(
                        proxy_type, proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass,
                    )
                else:
                    sock = socket.socket(family, socktype, proto)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if has_timeout(self.timeout):
                    sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))

                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

                # Python 3.3 compatibility: emulate the check_hostname behavior
                if not hasattr(self._context, "check_hostname") and not self.disable_ssl_certificate_validation:
                    try:
                        ssl.match_hostname(self.sock.getpeercert(), self.host)
                    except Exception:
                        self.sock.shutdown(socket.SHUT_RDWR)
                        self.sock.close()
                        raise

                if self.debuglevel > 0:
                    print("connect: ({0}, {1})".format(self.host, self.port))
                    if use_proxy:
                        print(
                            "proxy: {0}".format(
                                str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers,))
                            )
                        )
            except (ssl.SSLError, ssl.CertificateError) as e:
                if sock:
                    sock.close()
                if self.sock:
                    self.sock.close()
                self.sock = None
                raise
            except (socket.timeout, socket.gaierror):
                raise
            except socket.error as e:
                socket_err = e
                if self.debuglevel > 0:
                    print("connect fail: ({0}, {1})".format(self.host, self.port))
                    if use_proxy:
                        print(
                            "proxy: {0}".format(
                                str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers,))
                            )
                        )
                if self.sock:
                    self.sock.close()
                self.sock = None
                continue
            break
        if not self.sock:
            raise socket_err


SCHEME_TO_CONNECTION = {
    "http": HTTPConnectionWithTimeout,
    "https": HTTPSConnectionWithTimeout,
}


class Http(object):
    """An HTTP client that handles:

    - all methods
    - caching
    - ETags
    - compression,
    - HTTPS
    - Basic
    - Digest
    - WSSE

    and more.
    """

    def __init__(
        self,
        cache=None,
        timeout=None,
        proxy_info=proxy_info_from_environment,
        ca_certs=None,
        disable_ssl_certificate_validation=False,
        tls_maximum_version=None,
        tls_minimum_version=None,
    ):
        """If 'cache' is a string then it is used as a directory name for
        a disk cache. Otherwise it must be an object that supports the
        same interface as FileCache.

        All timeouts are in seconds. If None is passed for timeout
        then Python's default timeout for sockets will be used. See
        for example the docs of socket.setdefaulttimeout():
        http://docs.python.org/library/socket.html#socket.setdefaulttimeout

        `proxy_info` may be:
          - a callable that takes the http scheme ('http' or 'https') and
            returns a ProxyInfo instance per request. By default, uses
            proxy_info_from_environment.
          - a ProxyInfo instance (static proxy config).
          - None (proxy disabled).

        ca_certs is the path of a file containing root CA certificates for SSL
        server certificate validation.  By default, a CA cert file bundled with
        httplib2 is used.

        If disable_ssl_certificate_validation is true, SSL cert validation will
        not be performed.

        tls_maximum_version / tls_minimum_version require Python 3.7+ /
        OpenSSL 1.1.0g+. A value of "TLSv1_3" requires OpenSSL 1.1.1+.
        """
        self.proxy_info = proxy_info
        self.ca_certs = ca_certs
        self.disable_ssl_certificate_validation = disable_ssl_certificate_validation
        self.tls_maximum_version = tls_maximum_version
        self.tls_minimum_version = tls_minimum_version
        # Map domain name to an httplib connection
        self.connections = {}
        # The location of the cache, for now a directory
        # where cached responses are held.
        if cache and isinstance(cache, str):
            self.cache = FileCache(cache)
        else:
            self.cache = cache

        # Name/password
        self.credentials = Credentials()

        # Key/cert
        self.certificates = KeyCerts()

        # authorization objects
        self.authorizations = []

        # If set to False then no redirects are followed, even safe ones.
        self.follow_redirects = True

        self.redirect_codes = REDIRECT_CODES

        # Which HTTP methods do we apply optimistic concurrency to, i.e.
        # which methods get an "if-match:" etag header added to them.
        self.optimistic_concurrency_methods = ["PUT", "PATCH"]

        self.safe_methods = list(SAFE_METHODS)

        # If 'follow_redirects' is True, and this is set to True then
        # all redirecs are followed, including unsafe ones.
        self.follow_all_redirects = False

        self.ignore_etag = False

        self.force_exception_to_status_code = False

        self.timeout = timeout

        # Keep Authorization: headers on a redirect.
        self.forward_authorization_headers = False

    def close(self):
        """Close persistent connections, clear sensitive data.
        Not thread-safe, requires external synchronization against concurrent requests.
        """
        existing, self.connections = self.connections, {}
        for _, c in existing.items():
            c.close()
        self.certificates.clear()
        self.clear_credentials()

    def __getstate__(self):
        state_dict = copy.copy(self.__dict__)
        # In case request is augmented by some foreign object such as
        # credentials which handle auth
        if "request" in state_dict:
            del state_dict["request"]
        if "connections" in state_dict:
            del state_dict["connections"]
        return state_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.connections = {}

    def _auth_from_challenge(self, host, request_uri, headers, response, content):
        """A generator that creates Authorization objects
           that can be applied to requests.
        """
        challenges = auth._parse_www_authenticate(response, "www-authenticate")
        for cred in self.credentials.iter(host):
            for scheme in AUTH_SCHEME_ORDER:
                if scheme in challenges:
                    yield AUTH_SCHEME_CLASSES[scheme](cred, host, request_uri, headers, response, content, self)

    def add_credentials(self, name, password, domain=""):
        """Add a name and password that will be used
        any time a request requires authentication."""
        self.credentials.add(name, password, domain)

    def add_certificate(self, key, cert, domain, password=None):
        """Add a key and cert that will be used
        any time a request requires authentication."""
        self.certificates.add(key, cert, domain, password)

    def clear_credentials(self):
        """Remove all the names and passwords
        that are used for authentication"""
        self.credentials.clear()
        self.authorizations = []

    def _conn_request(self, conn, request_uri, method, body, headers):
        i = 0
        seen_bad_status_line = False
        while i < RETRIES:
            i += 1
            try:
                if conn.sock is None:
                    conn.connect()
                conn.request(method, request_uri, body, headers)
            except socket.timeout:
                conn.close()
                raise
            except socket.gaierror:
                conn.close()
                raise ServerNotFoundError("Unable to find the server at %s" % conn.host)
            except socket.error as e:
                errno_ = _errno_from_exception(e)
                if errno_ in (errno.ENETUNREACH, errno.EADDRNOTAVAIL) and i < RETRIES:
                    continue  # retry on potentially transient errors
                raise
            except http.client.HTTPException:
                if conn.sock is None:
                    if i < RETRIES - 1:
                        conn.close()
                        conn.connect()
                        continue
                    else:
                        conn.close()
                        raise
                if i < RETRIES - 1:
                    conn.close()
                    conn.connect()
                    continue
                # Just because the server closed the connection doesn't apparently mean
                # that the server didn't send a response.
                pass
            try:
                response = conn.getresponse()
            except (http.client.BadStatusLine, http.client.ResponseNotReady):
                # If we get a BadStatusLine on the first try then that means
                # the connection just went stale, so retry regardless of the
                # number of RETRIES set.
                if not seen_bad_status_line and i == 1:
                    i = 0
                    seen_bad_status_line = True
                    conn.close()
                    conn.connect()
                    continue
                else:
                    conn.close()
                    raise
            except socket.timeout:
                raise
            except (socket.error, http.client.HTTPException):
                conn.close()
                if i == 0:
                    conn.close()
                    conn.connect()
                    continue
                else:
                    raise
            else:
                content = b""
                if method == "HEAD":
                    conn.close()
                else:
                    content = response.read()
                response = Response(response)
                if method != "HEAD":
                    content = _decompressContent(response, content)

            break
        return (response, content)

    def _request(
        self, conn, host, absolute_uri, request_uri, method, body, headers, redirections, cachekey,
    ):
        """Do the actual request using the connection object
        and also follow one level of redirects if necessary"""

        auths = [(auth.depth(request_uri), auth) for auth in self.authorizations if auth.inscope(host, request_uri)]
        auth = auths and sorted(auths)[0][1] or None
        if auth:
            auth.request(method, request_uri, headers, body)

        (response, content) = self._conn_request(conn, request_uri, method, body, headers)

        if auth:
            if auth.response(response, body):
                auth.request(method, request_uri, headers, body)
                (response, content) = self._conn_request(conn, request_uri, method, body, headers)
                response._stale_digest = 1

        if response.status == 401:
            for authorization in self._auth_from_challenge(host, request_uri, headers, response, content):
                authorization.request(method, request_uri, headers, body)
                (response, content) = self._conn_request(conn, request_uri, method, body, headers)
                if response.status != 401:
                    self.authorizations.append(authorization)
                    authorization.response(response, body)
                    break

        if self.follow_all_redirects or method in self.safe_methods or response.status in (303, 308):
            if self.follow_redirects and response.status in self.redirect_codes:
                # Pick out the location header and basically start from the beginning
                # remembering first to strip the ETag header and decrement our 'depth'
                if redirections:
                    if "location" not in response and response.status != 300:
                        raise RedirectMissingLocation(
                            _("Redirected but the response is missing a Location: header."), response, content,
                        )
                    # Fix-up relative redirects (which violate an RFC 2616 MUST)
                    if "location" in response:
                        location = response["location"]
                        (scheme, authority, path, query, fragment) = parse_uri(location)
                        if authority == None:
                            response["location"] = urllib.parse.urljoin(absolute_uri, location)
                    if response.status == 308 or (response.status == 301 and (method in self.safe_methods)):
                        response["-x-permanent-redirect-url"] = response["location"]
                        if "content-location" not in response:
                            response["content-location"] = absolute_uri
                        _updateCache(headers, response, content, self.cache, cachekey)
                    if "if-none-match" in headers:
                        del headers["if-none-match"]
                    if "if-modified-since" in headers:
                        del headers["if-modified-since"]
                    if "authorization" in headers and not self.forward_authorization_headers:
                        del headers["authorization"]
                    if "location" in response:
                        location = response["location"]
                        old_response = copy.deepcopy(response)
                        if "content-location" not in old_response:
                            old_response["content-location"] = absolute_uri
                        redirect_method = method
                        if response.status in [302, 303]:
                            redirect_method = "GET"
                            body = None
                        (response, content) = self.request(
                            location, method=redirect_method, body=body, headers=headers, redirections=redirections - 1,
                        )
                        response.previous = old_response
                else:
                    raise RedirectLimit(
                        "Redirected more times than redirection_limit allows.", response, content,
                    )
            elif response.status in [200, 203] and method in self.safe_methods:
                # Don't cache 206's since we aren't going to handle byte range requests
                if "content-location" not in response:
                    response["content-location"] = absolute_uri
                _updateCache(headers, response, content, self.cache, cachekey)

        return (response, content)

    def _normalize_headers(self, headers):
        return _normalize_headers(headers)

    # Need to catch and rebrand some exceptions
    # Then need to optionally turn all exceptions into status codes
    # including all socket.* and httplib.* exceptions.

    def request(
        self, uri, method="GET", body=None, headers=None, redirections=DEFAULT_MAX_REDIRECTS, connection_type=None,
    ):
        """ Performs a single HTTP request.
The 'uri' is the URI of the HTTP resource and can begin
with either 'http' or 'https'. The value of 'uri' must be an absolute URI.

The 'method' is the HTTP method to perform, such as GET, POST, DELETE, etc.
There is no restriction on the methods allowed.

The 'body' is the entity body to be sent with the request. It is a string
object.

Any extra headers that are to be sent with the request should be provided in the
'headers' dictionary.

The maximum number of redirect to follow before raising an
exception is 'redirections. The default is 5.

The return value is a tuple of (response, content), the first
being and instance of the 'Response' class, the second being
a string that contains the response entity body.
        """
        conn_key = ""

        try:
            if headers is None:
                headers = {}
            else:
                headers = self._normalize_headers(headers)

            if "user-agent" not in headers:
                headers["user-agent"] = "Python-httplib2/%s (gzip)" % __version__

            uri = iri2uri(uri)
            # Prevent CWE-75 space injection to manipulate request via part of uri.
            # Prevent CWE-93 CRLF injection to modify headers via part of uri.
            uri = uri.replace(" ", "%20").replace("\r", "%0D").replace("\n", "%0A")

            (scheme, authority, request_uri, defrag_uri) = urlnorm(uri)

            conn_key = scheme + ":" + authority
            conn = self.connections.get(conn_key)
            if conn is None:
                if not connection_type:
                    connection_type = SCHEME_TO_CONNECTION[scheme]
                certs = list(self.certificates.iter(authority))
                if issubclass(connection_type, HTTPSConnectionWithTimeout):
                    if certs:
                        conn = self.connections[conn_key] = connection_type(
                            authority,
                            key_file=certs[0][0],
                            cert_file=certs[0][1],
                            timeout=self.timeout,
                            proxy_info=self.proxy_info,
                            ca_certs=self.ca_certs,
                            disable_ssl_certificate_validation=self.disable_ssl_certificate_validation,
                            tls_maximum_version=self.tls_maximum_version,
                            tls_minimum_version=self.tls_minimum_version,
                            key_password=certs[0][2],
                        )
                    else:
                        conn = self.connections[conn_key] = connection_type(
                            authority,
                            timeout=self.timeout,
                            proxy_info=self.proxy_info,
                            ca_certs=self.ca_certs,
                            disable_ssl_certificate_validation=self.disable_ssl_certificate_validation,
                            tls_maximum_version=self.tls_maximum_version,
                            tls_minimum_version=self.tls_minimum_version,
                        )
                else:
                    conn = self.connections[conn_key] = connection_type(
                        authority, timeout=self.timeout, proxy_info=self.proxy_info
                    )
                conn.set_debuglevel(debuglevel)

            if "range" not in headers and "accept-encoding" not in headers:
                headers["accept-encoding"] = "gzip, deflate"

            info = email.message.Message()
            cachekey = None
            cached_value = None
            if self.cache:
                cachekey = defrag_uri
                cached_value = self.cache.get(cachekey)
                if cached_value:
                    try:
                        info, content = cached_value.split(b"\r\n\r\n", 1)
                        info = email.message_from_bytes(info)
                        for k, v in info.items():
                            if v.startswith("=?") and v.endswith("?="):
                                info.replace_header(k, str(*email.header.decode_header(v)[0]))
                    except (IndexError, ValueError):
                        self.cache.delete(cachekey)
                        cachekey = None
                        cached_value = None

            if (
                method in self.optimistic_concurrency_methods
                and self.cache
                and "etag" in info
                and not self.ignore_etag
                and "if-match" not in headers
            ):
                # http://www.w3.org/1999/04/Editing/
                headers["if-match"] = info["etag"]

            # https://tools.ietf.org/html/rfc7234
            # A cache MUST invalidate the effective Request URI as well as [...] Location and Content-Location
            # when a non-error status code is received in response to an unsafe request method.
            if self.cache and cachekey and method not in self.safe_methods:
                self.cache.delete(cachekey)

            # Check the vary header in the cache to see if this request
            # matches what varies in the cache.
            if method in self.safe_methods and "vary" in info:
                vary = info["vary"]
                vary_headers = vary.lower().replace(" ", "").split(",")
                for header in vary_headers:
                    key = "-varied-%s" % header
                    value = info[key]
                    if headers.get(header, None) != value:
                        cached_value = None
                        break

            if (
                self.cache
                and cached_value
                and (method in self.safe_methods or info["status"] == "308")
                and "range" not in headers
            ):
                redirect_method = method
                if info["status"] not in ("307", "308"):
                    redirect_method = "GET"
                if "-x-permanent-redirect-url" in info:
                    # Should cached permanent redirects be counted in our redirection count? For now, yes.
                    if redirections <= 0:
                        raise RedirectLimit(
                            "Redirected more times than redirection_limit allows.", {}, "",
                        )
                    (response, new_content) = self.request(
                        info["-x-permanent-redirect-url"],
                        method=redirect_method,
                        headers=headers,
                        redirections=redirections - 1,
                    )
                    response.previous = Response(info)
                    response.previous.fromcache = True
                else:
                    # Determine our course of action:
                    #   Is the cached entry fresh or stale?
                    #   Has the client requested a non-cached response?
                    #
                    # There seems to be three possible answers:
                    # 1. [FRESH] Return the cache entry w/o doing a GET
                    # 2. [STALE] Do the GET (but add in cache validators if available)
                    # 3. [TRANSPARENT] Do a GET w/o any cache validators (Cache-Control: no-cache) on the request
                    entry_disposition = _entry_disposition(info, headers)

                    if entry_disposition == "FRESH":
                        response = Response(info)
                        response.fromcache = True
                        return (response, content)

                    if entry_disposition == "STALE":
                        if "etag" in info and not self.ignore_etag and not "if-none-match" in headers:
                            headers["if-none-match"] = info["etag"]
                        if "last-modified" in info and not "last-modified" in headers:
                            headers["if-modified-since"] = info["last-modified"]
                    elif entry_disposition == "TRANSPARENT":
                        pass

                    (response, new_content) = self._request(
                        conn, authority, uri, request_uri, method, body, headers, redirections, cachekey,
                    )

                if response.status == 304 and method == "GET":
                    # Rewrite the cache entry with the new end-to-end headers
                    # Take all headers that are in response
                    # and overwrite their values in info.
                    # unless they are hop-by-hop, or are listed in the connection header.

                    for key in _get_end2end_headers(response):
                        info[key] = response[key]
                    merged_response = Response(info)
                    if hasattr(response, "_stale_digest"):
                        merged_response._stale_digest = response._stale_digest
                    _updateCache(headers, merged_response, content, self.cache, cachekey)
                    response = merged_response
                    response.status = 200
                    response.fromcache = True

                elif response.status == 200:
                    content = new_content
                else:
                    self.cache.delete(cachekey)
                    content = new_content
            else:
                cc = _parse_cache_control(headers)
                if "only-if-cached" in cc:
                    info["status"] = "504"
                    response = Response(info)
                    content = b""
                else:
                    (response, content) = self._request(
                        conn, authority, uri, request_uri, method, body, headers, redirections, cachekey,
                    )
        except Exception as e:
            is_timeout = isinstance(e, socket.timeout)
            if is_timeout:
                conn = self.connections.pop(conn_key, None)
                if conn:
                    conn.close()

            if self.force_exception_to_status_code:
                if isinstance(e, HttpLib2ErrorWithResponse):
                    response = e.response
                    content = e.content
                    response.status = 500
                    response.reason = str(e)
                elif isinstance(e, socket.timeout):
                    content = b"Request Timeout"
                    response = Response({"content-type": "text/plain", "status": "408", "content-length": len(content),})
                    response.reason = "Request Timeout"
                else:
                    content = str(e).encode("utf-8")
                    response = Response({"content-type": "text/plain", "status": "400", "content-length": len(content),})
                    response.reason = "Bad Request"
            else:
                raise

        return (response, content)


class Response(dict):
    """An object more like email.message than httplib.HTTPResponse."""

    """Is this response from our local cache"""
    fromcache = False
    """HTTP protocol version used by server.

    10 for HTTP/1.0, 11 for HTTP/1.1.
    """
    version = 11

    "Status code returned by server. "
    status = 200
    """Reason phrase returned by server."""
    reason = "Ok"

    previous = None

    def __init__(self, info):
        # info is either an email.message or
        # an httplib.HTTPResponse object.
        if isinstance(info, http.client.HTTPResponse):
            for key, value in info.getheaders():
                key = key.lower()
                prev = self.get(key)
                if prev is not None:
                    value = ", ".join((prev, value))
                self[key] = value
            self.status = info.status
            self["status"] = str(self.status)
            self.reason = info.reason
            self.version = info.version
        elif isinstance(info, email.message.Message):
            for key, value in list(info.items()):
                self[key.lower()] = value
            self.status = int(self["status"])
        else:
            for key, value in info.items():
                self[key.lower()] = value
            self.status = int(self.get("status", self.status))

    def __getattr__(self, name):
        if name == "dict":
            return self
        else:
            raise AttributeError(name)
