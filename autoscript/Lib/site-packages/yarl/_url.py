import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from ipaddress import ip_address
from urllib.parse import SplitResult, parse_qsl, quote, urljoin, urlsplit, urlunsplit

import idna
from multidict import MultiDict, MultiDictProxy

from ._quoting import _Quoter, _Unquoter

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

sentinel = object()


def rewrite_module(obj: object) -> object:
    obj.__module__ = "yarl"
    return obj


class cached_property:
    """Use as a class method decorator.  It operates almost exactly like
    the Python `@property` decorator, but it puts the result of the
    method it decorates into the instance dict after the first call,
    effectively replacing the function it decorates with an instance
    variable.  It is, in Python parlance, a data descriptor.

    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        try:
            self.__doc__ = wrapped.__doc__
        except AttributeError:  # pragma: no cover
            self.__doc__ = ""
        self.name = wrapped.__name__

    def __get__(self, inst, owner, _sentinel=sentinel):
        if inst is None:
            return self
        val = inst._cache.get(self.name, _sentinel)
        if val is not _sentinel:
            return val
        val = self.wrapped(inst)
        inst._cache[self.name] = val
        return val

    def __set__(self, inst, value):
        raise AttributeError("cached property is read-only")


@rewrite_module
class URL:
    # Don't derive from str
    # follow pathlib.Path design
    # probably URL will not suffer from pathlib problems:
    # it's intended for libraries like aiohttp,
    # not to be passed into standard library functions like os.open etc.

    # URL grammar (RFC 3986)
    # pct-encoded = "%" HEXDIG HEXDIG
    # reserved    = gen-delims / sub-delims
    # gen-delims  = ":" / "/" / "?" / "#" / "[" / "]" / "@"
    # sub-delims  = "!" / "$" / "&" / "'" / "(" / ")"
    #             / "*" / "+" / "," / ";" / "="
    # unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"
    # URI         = scheme ":" hier-part [ "?" query ] [ "#" fragment ]
    # hier-part   = "//" authority path-abempty
    #             / path-absolute
    #             / path-rootless
    #             / path-empty
    # scheme      = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
    # authority   = [ userinfo "@" ] host [ ":" port ]
    # userinfo    = *( unreserved / pct-encoded / sub-delims / ":" )
    # host        = IP-literal / IPv4address / reg-name
    # IP-literal = "[" ( IPv6address / IPvFuture  ) "]"
    # IPvFuture  = "v" 1*HEXDIG "." 1*( unreserved / sub-delims / ":" )
    # IPv6address =                            6( h16 ":" ) ls32
    #             /                       "::" 5( h16 ":" ) ls32
    #             / [               h16 ] "::" 4( h16 ":" ) ls32
    #             / [ *1( h16 ":" ) h16 ] "::" 3( h16 ":" ) ls32
    #             / [ *2( h16 ":" ) h16 ] "::" 2( h16 ":" ) ls32
    #             / [ *3( h16 ":" ) h16 ] "::"    h16 ":"   ls32
    #             / [ *4( h16 ":" ) h16 ] "::"              ls32
    #             / [ *5( h16 ":" ) h16 ] "::"              h16
    #             / [ *6( h16 ":" ) h16 ] "::"
    # ls32        = ( h16 ":" h16 ) / IPv4address
    #             ; least-significant 32 bits of address
    # h16         = 1*4HEXDIG
    #             ; 16 bits of address represented in hexadecimal
    # IPv4address = dec-octet "." dec-octet "." dec-octet "." dec-octet
    # dec-octet   = DIGIT                 ; 0-9
    #             / %x31-39 DIGIT         ; 10-99
    #             / "1" 2DIGIT            ; 100-199
    #             / "2" %x30-34 DIGIT     ; 200-249
    #             / "25" %x30-35          ; 250-255
    # reg-name    = *( unreserved / pct-encoded / sub-delims )
    # port        = *DIGIT
    # path          = path-abempty    ; begins with "/" or is empty
    #               / path-absolute   ; begins with "/" but not "//"
    #               / path-noscheme   ; begins with a non-colon segment
    #               / path-rootless   ; begins with a segment
    #               / path-empty      ; zero characters
    # path-abempty  = *( "/" segment )
    # path-absolute = "/" [ segment-nz *( "/" segment ) ]
    # path-noscheme = segment-nz-nc *( "/" segment )
    # path-rootless = segment-nz *( "/" segment )
    # path-empty    = 0<pchar>
    # segment       = *pchar
    # segment-nz    = 1*pchar
    # segment-nz-nc = 1*( unreserved / pct-encoded / sub-delims / "@" )
    #               ; non-zero-length segment without any colon ":"
    # pchar         = unreserved / pct-encoded / sub-delims / ":" / "@"
    # query       = *( pchar / "/" / "?" )
    # fragment    = *( pchar / "/" / "?" )
    # URI-reference = URI / relative-ref
    # relative-ref  = relative-part [ "?" query ] [ "#" fragment ]
    # relative-part = "//" authority path-abempty
    #               / path-absolute
    #               / path-noscheme
    #               / path-empty
    # absolute-URI  = scheme ":" hier-part [ "?" query ]
    __slots__ = ("_cache", "_val")

    _QUOTER = _Quoter(requote=False)
    _REQUOTER = _Quoter()
    _PATH_QUOTER = _Quoter(safe="@:", protected="/+", requote=False)
    _PATH_REQUOTER = _Quoter(safe="@:", protected="/+")
    _QUERY_QUOTER = _Quoter(safe="?/:@", protected="=+&;", qs=True, requote=False)
    _QUERY_REQUOTER = _Quoter(safe="?/:@", protected="=+&;", qs=True)
    _QUERY_PART_QUOTER = _Quoter(safe="?/:@", qs=True, requote=False)
    _FRAGMENT_QUOTER = _Quoter(safe="?/:@", requote=False)
    _FRAGMENT_REQUOTER = _Quoter(safe="?/:@")

    _UNQUOTER = _Unquoter()
    _PATH_UNQUOTER = _Unquoter(unsafe="+")
    _QS_UNQUOTER = _Unquoter(qs=True)

    def __new__(cls, val="", *, encoded=False, strict=None):
        if strict is not None:  # pragma: no cover
            warnings.warn("strict parameter is ignored")
        if type(val) is cls:
            return val
        if type(val) is str:
            val = urlsplit(val)
        elif type(val) is SplitResult:
            if not encoded:
                raise ValueError("Cannot apply decoding to SplitResult")
        elif isinstance(val, str):
            val = urlsplit(str(val))
        else:
            raise TypeError("Constructor parameter should be str")

        if not encoded:
            if not val[1]:  # netloc
                netloc = ""
                host = ""
            else:
                host = val.hostname
                if host is None:
                    raise ValueError("Invalid URL: host is required for absolute urls")

                try:
                    port = val.port
                except ValueError as e:
                    raise ValueError(
                        "Invalid URL: port can't be converted to integer"
                    ) from e

                netloc = cls._make_netloc(
                    val.username, val.password, host, port, encode=True, requote=True
                )
            path = cls._PATH_REQUOTER(val[2])
            if netloc:
                path = cls._normalize_path(path)

            cls._validate_authority_uri_abs_path(host=host, path=path)
            query = cls._QUERY_REQUOTER(val[3])
            fragment = cls._FRAGMENT_REQUOTER(val[4])
            val = SplitResult(val[0], netloc, path, query, fragment)

        self = object.__new__(cls)
        self._val = val
        self._cache = {}
        return self

    @classmethod
    def build(
        cls,
        *,
        scheme="",
        authority="",
        user=None,
        password=None,
        host="",
        port=None,
        path="",
        query=None,
        query_string="",
        fragment="",
        encoded=False,
    ):
        """Creates and returns a new URL"""

        if authority and (user or password or host or port):
            raise ValueError(
                'Can\'t mix "authority" with "user", "password", "host" or "port".'
            )
        if port and not host:
            raise ValueError('Can\'t build URL with "port" but without "host".')
        if query and query_string:
            raise ValueError('Only one of "query" or "query_string" should be passed')
        if (
            scheme is None
            or authority is None
            or path is None
            or query_string is None
            or fragment is None
        ):
            raise TypeError(
                'NoneType is illegal for "scheme", "authority", "path", '
                '"query_string", and "fragment" args, use empty string instead.'
            )

        if authority:
            if encoded:
                netloc = authority
            else:
                tmp = SplitResult("", authority, "", "", "")
                netloc = cls._make_netloc(
                    tmp.username, tmp.password, tmp.hostname, tmp.port, encode=True
                )
        elif not user and not password and not host and not port:
            netloc = ""
        else:
            netloc = cls._make_netloc(
                user, password, host, port, encode=not encoded, encode_host=not encoded
            )
        if not encoded:
            path = cls._PATH_QUOTER(path)
            if netloc:
                path = cls._normalize_path(path)

            cls._validate_authority_uri_abs_path(host=host, path=path)
            query_string = cls._QUERY_QUOTER(query_string)
            fragment = cls._FRAGMENT_QUOTER(fragment)

        url = cls(
            SplitResult(scheme, netloc, path, query_string, fragment), encoded=True
        )

        if query:
            return url.with_query(query)
        else:
            return url

    def __init_subclass__(cls):
        raise TypeError(f"Inheriting a class {cls!r} from URL is forbidden")

    def __str__(self):
        val = self._val
        if not val.path and self.is_absolute() and (val.query or val.fragment):
            val = val._replace(path="/")
        return urlunsplit(val)

    def __repr__(self):
        return f"{self.__class__.__name__}('{str(self)}')"

    def __bytes__(self):
        return str(self).encode("ascii")

    def __eq__(self, other):
        if not type(other) is URL:
            return NotImplemented

        val1 = self._val
        if not val1.path and self.is_absolute():
            val1 = val1._replace(path="/")

        val2 = other._val
        if not val2.path and other.is_absolute():
            val2 = val2._replace(path="/")

        return val1 == val2

    def __hash__(self):
        ret = self._cache.get("hash")
        if ret is None:
            val = self._val
            if not val.path and self.is_absolute():
                val = val._replace(path="/")
            ret = self._cache["hash"] = hash(val)
        return ret

    def __le__(self, other):
        if not type(other) is URL:
            return NotImplemented
        return self._val <= other._val

    def __lt__(self, other):
        if not type(other) is URL:
            return NotImplemented
        return self._val < other._val

    def __ge__(self, other):
        if not type(other) is URL:
            return NotImplemented
        return self._val >= other._val

    def __gt__(self, other):
        if not type(other) is URL:
            return NotImplemented
        return self._val > other._val

    def __truediv__(self, name):
        name = self._PATH_QUOTER(name)
        if name.startswith("/"):
            raise ValueError(
                f"Appending path {name!r} starting from slash is forbidden"
            )
        path = self._val.path
        if path == "/":
            new_path = "/" + name
        elif not path and not self.is_absolute():
            new_path = name
        else:
            parts = path.rstrip("/").split("/")
            parts.append(name)
            new_path = "/".join(parts)
        if self.is_absolute():
            new_path = self._normalize_path(new_path)
        return URL(
            self._val._replace(path=new_path, query="", fragment=""), encoded=True
        )

    def __mod__(self, query):
        return self.update_query(query)

    def __bool__(self) -> bool:
        return bool(
            self._val.netloc or self._val.path or self._val.query or self._val.fragment
        )

    def __getstate__(self):
        return (self._val,)

    def __setstate__(self, state):
        if state[0] is None and isinstance(state[1], dict):
            # default style pickle
            self._val = state[1]["_val"]
        else:
            self._val, *unused = state
        self._cache = {}

    def is_absolute(self):
        """A check for absolute URLs.

        Return True for absolute ones (having scheme or starting
        with //), False otherwise.

        """
        return self.raw_host is not None

    def is_default_port(self):
        """A check for default port.

        Return True if port is default for specified scheme,
        e.g. 'http://python.org' or 'http://python.org:80', False
        otherwise.

        """
        if self.port is None:
            return False
        default = DEFAULT_PORTS.get(self.scheme)
        if default is None:
            return False
        return self.port == default

    def origin(self):
        """Return an URL with scheme, host and port parts only.

        user, password, path, query and fragment are removed.

        """
        # TODO: add a keyword-only option for keeping user/pass maybe?
        if not self.is_absolute():
            raise ValueError("URL should be absolute")
        if not self._val.scheme:
            raise ValueError("URL should have scheme")
        v = self._val
        netloc = self._make_netloc(None, None, v.hostname, v.port)
        val = v._replace(netloc=netloc, path="", query="", fragment="")
        return URL(val, encoded=True)

    def relative(self):
        """Return a relative part of the URL.

        scheme, user, password, host and port are removed.

        """
        if not self.is_absolute():
            raise ValueError("URL should be absolute")
        val = self._val._replace(scheme="", netloc="")
        return URL(val, encoded=True)

    @property
    def scheme(self):
        """Scheme for absolute URLs.

        Empty string for relative URLs or URLs starting with //

        """
        return self._val.scheme

    @property
    def raw_authority(self):
        """Encoded authority part of URL.

        Empty string for relative URLs.

        """
        return self._val.netloc

    @cached_property
    def authority(self):
        """Decoded authority part of URL.

        Empty string for relative URLs.

        """
        return self._make_netloc(
            self.user, self.password, self.host, self.port, encode_host=False
        )

    @property
    def raw_user(self):
        """Encoded user part of URL.

        None if user is missing.

        """
        # not .username
        ret = self._val.username
        if not ret:
            return None
        return ret

    @cached_property
    def user(self):
        """Decoded user part of URL.

        None if user is missing.

        """
        return self._UNQUOTER(self.raw_user)

    @property
    def raw_password(self):
        """Encoded password part of URL.

        None if password is missing.

        """
        return self._val.password

    @cached_property
    def password(self):
        """Decoded password part of URL.

        None if password is missing.

        """
        return self._UNQUOTER(self.raw_password)

    @property
    def raw_host(self):
        """Encoded host part of URL.

        None for relative URLs.

        """
        # Use host instead of hostname for sake of shortness
        # May add .hostname prop later
        return self._val.hostname

    @cached_property
    def host(self):
        """Decoded host part of URL.

        None for relative URLs.

        """
        raw = self.raw_host
        if raw is None:
            return None
        if "%" in raw:
            # Hack for scoped IPv6 addresses like
            # fe80::2%Проверка
            # presence of '%' sign means only IPv6 address, so idna is useless.
            return raw
        return _idna_decode(raw)

    @property
    def port(self):
        """Port part of URL, with scheme-based fallback.

        None for relative URLs or URLs without explicit port and
        scheme without default port substitution.

        """
        return self._val.port or DEFAULT_PORTS.get(self._val.scheme)

    @property
    def explicit_port(self):
        """Port part of URL, without scheme-based fallback.

        None for relative URLs or URLs without explicit port.

        """
        return self._val.port

    @property
    def raw_path(self):
        """Encoded path of URL.

        / for absolute URLs without path part.

        """
        ret = self._val.path
        if not ret and self.is_absolute():
            ret = "/"
        return ret

    @cached_property
    def path(self):
        """Decoded path of URL.

        / for absolute URLs without path part.

        """
        return self._PATH_UNQUOTER(self.raw_path)

    @cached_property
    def query(self):
        """A MultiDictProxy representing parsed query parameters in decoded
        representation.

        Empty value if URL has no query part.

        """
        ret = MultiDict(parse_qsl(self.raw_query_string, keep_blank_values=True))
        return MultiDictProxy(ret)

    @property
    def raw_query_string(self):
        """Encoded query part of URL.

        Empty string if query is missing.

        """
        return self._val.query

    @cached_property
    def query_string(self):
        """Decoded query part of URL.

        Empty string if query is missing.

        """
        return self._QS_UNQUOTER(self.raw_query_string)

    @cached_property
    def path_qs(self):
        """Decoded path of URL with query."""
        if not self.query_string:
            return self.path
        return f"{self.path}?{self.query_string}"

    @cached_property
    def raw_path_qs(self):
        """Encoded path of URL with query."""
        if not self.raw_query_string:
            return self.raw_path
        return f"{self.raw_path}?{self.raw_query_string}"

    @property
    def raw_fragment(self):
        """Encoded fragment part of URL.

        Empty string if fragment is missing.

        """
        return self._val.fragment

    @cached_property
    def fragment(self):
        """Decoded fragment part of URL.

        Empty string if fragment is missing.

        """
        return self._UNQUOTER(self.raw_fragment)

    @cached_property
    def raw_parts(self):
        """A tuple containing encoded *path* parts.

        ('/',) for absolute URLs if *path* is missing.

        """
        path = self._val.path
        if self.is_absolute():
            if not path:
                parts = ["/"]
            else:
                parts = ["/"] + path[1:].split("/")
        else:
            if path.startswith("/"):
                parts = ["/"] + path[1:].split("/")
            else:
                parts = path.split("/")
        return tuple(parts)

    @cached_property
    def parts(self):
        """A tuple containing decoded *path* parts.

        ('/',) for absolute URLs if *path* is missing.

        """
        return tuple(self._UNQUOTER(part) for part in self.raw_parts)

    @cached_property
    def parent(self):
        """A new URL with last part of path removed and cleaned up query and
        fragment.

        """
        path = self.raw_path
        if not path or path == "/":
            if self.raw_fragment or self.raw_query_string:
                return URL(self._val._replace(query="", fragment=""), encoded=True)
            return self
        parts = path.split("/")
        val = self._val._replace(path="/".join(parts[:-1]), query="", fragment="")
        return URL(val, encoded=True)

    @cached_property
    def raw_name(self):
        """The last part of raw_parts."""
        parts = self.raw_parts
        if self.is_absolute():
            parts = parts[1:]
            if not parts:
                return ""
            else:
                return parts[-1]
        else:
            return parts[-1]

    @cached_property
    def name(self):
        """The last part of parts."""
        return self._UNQUOTER(self.raw_name)

    @cached_property
    def raw_suffix(self):
        name = self.raw_name
        i = name.rfind(".")
        if 0 < i < len(name) - 1:
            return name[i:]
        else:
            return ""

    @cached_property
    def suffix(self):
        return self._UNQUOTER(self.raw_suffix)

    @cached_property
    def raw_suffixes(self):
        name = self.raw_name
        if name.endswith("."):
            return ()
        name = name.lstrip(".")
        return tuple("." + suffix for suffix in name.split(".")[1:])

    @cached_property
    def suffixes(self):
        return tuple(self._UNQUOTER(suffix) for suffix in self.raw_suffixes)

    @staticmethod
    def _validate_authority_uri_abs_path(host, path):
        """Ensure that path in URL with authority starts with a leading slash.

        Raise ValueError if not.
        """
        if len(host) > 0 and len(path) > 0 and not path.startswith("/"):
            raise ValueError(
                "Path in a URL with authority should start with a slash ('/') if set"
            )

    @classmethod
    def _normalize_path(cls, path):
        # Drop '.' and '..' from path

        segments = path.split("/")
        resolved_path = []

        for seg in segments:
            if seg == "..":
                try:
                    resolved_path.pop()
                except IndexError:
                    # ignore any .. segments that would otherwise cause an
                    # IndexError when popped from resolved_path if
                    # resolving for rfc3986
                    pass
            elif seg == ".":
                continue
            else:
                resolved_path.append(seg)

        if segments[-1] in (".", ".."):
            # do some post-processing here.
            # if the last segment was a relative dir,
            # then we need to append the trailing '/'
            resolved_path.append("")

        return "/".join(resolved_path)

    @classmethod
    def _encode_host(cls, host, human=False):
        try:
            ip, sep, zone = host.partition("%")
            ip = ip_address(ip)
        except ValueError:
            host = host.lower()
            # IDNA encoding is slow,
            # skip it for ASCII-only strings
            # Don't move the check into _idna_encode() helper
            # to reduce the cache size
            if human or host.isascii():
                return host
            host = _idna_encode(host)
        else:
            host = ip.compressed
            if sep:
                host += "%" + zone
            if ip.version == 6:
                host = "[" + host + "]"
        return host

    @classmethod
    def _make_netloc(
        cls, user, password, host, port, encode=False, encode_host=True, requote=False
    ):
        quoter = cls._REQUOTER if requote else cls._QUOTER
        if encode_host:
            ret = cls._encode_host(host)
        else:
            ret = host
        if port:
            ret = ret + ":" + str(port)
        if password is not None:
            if not user:
                user = ""
            else:
                if encode:
                    user = quoter(user)
            if encode:
                password = quoter(password)
            user = user + ":" + password
        elif user and encode:
            user = quoter(user)
        if user:
            ret = user + "@" + ret
        return ret

    def with_scheme(self, scheme):
        """Return a new URL with scheme replaced."""
        # N.B. doesn't cleanup query/fragment
        if not isinstance(scheme, str):
            raise TypeError("Invalid scheme type")
        if not self.is_absolute():
            raise ValueError("scheme replacement is not allowed for relative URLs")
        return URL(self._val._replace(scheme=scheme.lower()), encoded=True)

    def with_user(self, user):
        """Return a new URL with user replaced.

        Autoencode user if needed.

        Clear user/password if user is None.

        """
        # N.B. doesn't cleanup query/fragment
        val = self._val
        if user is None:
            password = None
        elif isinstance(user, str):
            user = self._QUOTER(user)
            password = val.password
        else:
            raise TypeError("Invalid user type")
        if not self.is_absolute():
            raise ValueError("user replacement is not allowed for relative URLs")
        return URL(
            self._val._replace(
                netloc=self._make_netloc(user, password, val.hostname, val.port)
            ),
            encoded=True,
        )

    def with_password(self, password):
        """Return a new URL with password replaced.

        Autoencode password if needed.

        Clear password if argument is None.

        """
        # N.B. doesn't cleanup query/fragment
        if password is None:
            pass
        elif isinstance(password, str):
            password = self._QUOTER(password)
        else:
            raise TypeError("Invalid password type")
        if not self.is_absolute():
            raise ValueError("password replacement is not allowed for relative URLs")
        val = self._val
        return URL(
            self._val._replace(
                netloc=self._make_netloc(val.username, password, val.hostname, val.port)
            ),
            encoded=True,
        )

    def with_host(self, host):
        """Return a new URL with host replaced.

        Autoencode host if needed.

        Changing host for relative URLs is not allowed, use .join()
        instead.

        """
        # N.B. doesn't cleanup query/fragment
        if not isinstance(host, str):
            raise TypeError("Invalid host type")
        if not self.is_absolute():
            raise ValueError("host replacement is not allowed for relative URLs")
        if not host:
            raise ValueError("host removing is not allowed")
        val = self._val
        return URL(
            self._val._replace(
                netloc=self._make_netloc(val.username, val.password, host, val.port)
            ),
            encoded=True,
        )

    def with_port(self, port):
        """Return a new URL with port replaced.

        Clear port to default if None is passed.

        """
        # N.B. doesn't cleanup query/fragment
        if port is not None and not isinstance(port, int):
            raise TypeError(f"port should be int or None, got {type(port)}")
        if not self.is_absolute():
            raise ValueError("port replacement is not allowed for relative URLs")
        val = self._val
        return URL(
            self._val._replace(
                netloc=self._make_netloc(val.username, val.password, val.hostname, port)
            ),
            encoded=True,
        )

    def with_path(self, path, *, encoded=False):
        """Return a new URL with path replaced."""
        if not encoded:
            path = self._PATH_QUOTER(path)
            if self.is_absolute():
                path = self._normalize_path(path)
        if len(path) > 0 and path[0] != "/":
            path = "/" + path
        return URL(self._val._replace(path=path, query="", fragment=""), encoded=True)

    @classmethod
    def _query_seq_pairs(cls, quoter, pairs):
        for key, val in pairs:
            if isinstance(val, (list, tuple)):
                for v in val:
                    yield quoter(key) + "=" + quoter(cls._query_var(v))
            else:
                yield quoter(key) + "=" + quoter(cls._query_var(val))

    @staticmethod
    def _query_var(v):
        cls = type(v)
        if issubclass(cls, str):
            return v
        if issubclass(cls, float):
            if math.isinf(v):
                raise ValueError("float('inf') is not supported")
            if math.isnan(v):
                raise ValueError("float('nan') is not supported")
            return str(float(v))
        if issubclass(cls, int) and cls is not bool:
            return str(int(v))
        raise TypeError(
            "Invalid variable type: value "
            "should be str, int or float, got {!r} "
            "of type {}".format(v, cls)
        )

    def _get_str_query(self, *args, **kwargs):
        if kwargs:
            if len(args) > 0:
                raise ValueError(
                    "Either kwargs or single query parameter must be present"
                )
            query = kwargs
        elif len(args) == 1:
            query = args[0]
        else:
            raise ValueError("Either kwargs or single query parameter must be present")

        if query is None:
            query = ""
        elif isinstance(query, Mapping):
            quoter = self._QUERY_PART_QUOTER
            query = "&".join(self._query_seq_pairs(quoter, query.items()))
        elif isinstance(query, str):
            query = self._QUERY_QUOTER(query)
        elif isinstance(query, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Invalid query type: bytes, bytearray and memoryview are forbidden"
            )
        elif isinstance(query, Sequence):
            quoter = self._QUERY_PART_QUOTER
            # We don't expect sequence values if we're given a list of pairs
            # already; only mappings like builtin `dict` which can't have the
            # same key pointing to multiple values are allowed to use
            # `_query_seq_pairs`.
            query = "&".join(
                quoter(k) + "=" + quoter(self._query_var(v)) for k, v in query
            )
        else:
            raise TypeError(
                "Invalid query type: only str, mapping or "
                "sequence of (key, value) pairs is allowed"
            )

        return query

    def with_query(self, *args, **kwargs):
        """Return a new URL with query part replaced.

        Accepts any Mapping (e.g. dict, multidict.MultiDict instances)
        or str, autoencode the argument if needed.

        A sequence of (key, value) pairs is supported as well.

        It also can take an arbitrary number of keyword arguments.

        Clear query if None is passed.

        """
        # N.B. doesn't cleanup query/fragment

        new_query = self._get_str_query(*args, **kwargs)
        return URL(
            self._val._replace(path=self._val.path, query=new_query), encoded=True
        )

    def update_query(self, *args, **kwargs):
        """Return a new URL with query part updated."""
        s = self._get_str_query(*args, **kwargs)
        new_query = MultiDict(parse_qsl(s, keep_blank_values=True))
        query = MultiDict(self.query)
        query.update(new_query)

        return URL(self._val._replace(query=self._get_str_query(query)), encoded=True)

    def with_fragment(self, fragment):
        """Return a new URL with fragment replaced.

        Autoencode fragment if needed.

        Clear fragment to default if None is passed.

        """
        # N.B. doesn't cleanup query/fragment
        if fragment is None:
            raw_fragment = ""
        elif not isinstance(fragment, str):
            raise TypeError("Invalid fragment type")
        else:
            raw_fragment = self._FRAGMENT_QUOTER(fragment)
        if self.raw_fragment == raw_fragment:
            return self
        return URL(self._val._replace(fragment=raw_fragment), encoded=True)

    def with_name(self, name):
        """Return a new URL with name (last part of path) replaced.

        Query and fragment parts are cleaned up.

        Name is encoded if needed.

        """
        # N.B. DOES cleanup query/fragment
        if not isinstance(name, str):
            raise TypeError("Invalid name type")
        if "/" in name:
            raise ValueError("Slash in name is not allowed")
        name = self._PATH_QUOTER(name)
        if name in (".", ".."):
            raise ValueError(". and .. values are forbidden")
        parts = list(self.raw_parts)
        if self.is_absolute():
            if len(parts) == 1:
                parts.append(name)
            else:
                parts[-1] = name
            parts[0] = ""  # replace leading '/'
        else:
            parts[-1] = name
            if parts[0] == "/":
                parts[0] = ""  # replace leading '/'
        return URL(
            self._val._replace(path="/".join(parts), query="", fragment=""),
            encoded=True,
        )

    def with_suffix(self, suffix):
        """Return a new URL with suffix (file extension of name) replaced.

        Query and fragment parts are cleaned up.

        suffix is encoded if needed.
        """
        if not isinstance(suffix, str):
            raise TypeError("Invalid suffix type")
        if suffix and not suffix.startswith(".") or suffix == ".":
            raise ValueError(f"Invalid suffix {suffix!r}")
        name = self.raw_name
        if not name:
            raise ValueError(f"{self!r} has an empty name")
        old_suffix = self.raw_suffix
        if not old_suffix:
            name = name + suffix
        else:
            name = name[: -len(old_suffix)] + suffix
        return self.with_name(name)

    def join(self, url):
        """Join URLs

        Construct a full (“absolute”) URL by combining a “base URL”
        (self) with another URL (url).

        Informally, this uses components of the base URL, in
        particular the addressing scheme, the network location and
        (part of) the path, to provide missing components in the
        relative URL.

        """
        # See docs for urllib.parse.urljoin
        if not isinstance(url, URL):
            raise TypeError("url should be URL")
        return URL(urljoin(str(self), str(url)), encoded=True)

    def human_repr(self):
        """Return decoded human readable string for URL representation."""
        user = _human_quote(self.user, "#/:?@")
        password = _human_quote(self.password, "#/:?@")
        host = self.host
        if host:
            host = self._encode_host(self.host, human=True)
        path = _human_quote(self.path, "#?")
        query_string = "&".join(
            "{}={}".format(_human_quote(k, "#&+;="), _human_quote(v, "#&+;="))
            for k, v in self.query.items()
        )
        fragment = _human_quote(self.fragment, "")
        return urlunsplit(
            SplitResult(
                self.scheme,
                self._make_netloc(
                    user,
                    password,
                    host,
                    self._val.port,
                    encode_host=False,
                ),
                path,
                query_string,
                fragment,
            )
        )


def _human_quote(s, unsafe):
    if not s:
        return s
    for c in "%" + unsafe:
        if c in s:
            s = s.replace(c, f"%{ord(c):02X}")
    if s.isprintable():
        return s
    return "".join(c if c.isprintable() else quote(c) for c in s)


_MAXCACHE = 256


@functools.lru_cache(_MAXCACHE)
def _idna_decode(raw):
    try:
        return idna.decode(raw.encode("ascii"))
    except UnicodeError:  # e.g. '::1'
        return raw.encode("ascii").decode("idna")


@functools.lru_cache(_MAXCACHE)
def _idna_encode(host):
    try:
        return idna.encode(host, uts46=True).decode("ascii")
    except UnicodeError:
        return host.encode("idna").decode("ascii")


@rewrite_module
def cache_clear():
    _idna_decode.cache_clear()
    _idna_encode.cache_clear()


@rewrite_module
def cache_info():
    return {
        "idna_encode": _idna_encode.cache_info(),
        "idna_decode": _idna_decode.cache_info(),
    }


@rewrite_module
def cache_configure(*, idna_encode_size=_MAXCACHE, idna_decode_size=_MAXCACHE):
    global _idna_decode, _idna_encode

    _idna_encode = functools.lru_cache(idna_encode_size)(_idna_encode.__wrapped__)
    _idna_decode = functools.lru_cache(idna_decode_size)(_idna_decode.__wrapped__)
