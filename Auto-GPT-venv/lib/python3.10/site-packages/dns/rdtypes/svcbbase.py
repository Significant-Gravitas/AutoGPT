# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import base64
import enum
import io
import struct

import dns.enum
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
import dns.rdtypes.util
import dns.tokenizer
import dns.wire

# Until there is an RFC, this module is experimental and may be changed in
# incompatible ways.


class UnknownParamKey(dns.exception.DNSException):
    """Unknown SVCB ParamKey"""


class ParamKey(dns.enum.IntEnum):
    """SVCB ParamKey"""

    MANDATORY = 0
    ALPN = 1
    NO_DEFAULT_ALPN = 2
    PORT = 3
    IPV4HINT = 4
    ECH = 5
    IPV6HINT = 6

    @classmethod
    def _maximum(cls):
        return 65535

    @classmethod
    def _short_name(cls):
        return "SVCBParamKey"

    @classmethod
    def _prefix(cls):
        return "KEY"

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownParamKey


class Emptiness(enum.IntEnum):
    NEVER = 0
    ALWAYS = 1
    ALLOWED = 2


def _validate_key(key):
    force_generic = False
    if isinstance(key, bytes):
        # We decode to latin-1 so we get 0-255 as valid and do NOT interpret
        # UTF-8 sequences
        key = key.decode("latin-1")
    if isinstance(key, str):
        if key.lower().startswith("key"):
            force_generic = True
            if key[3:].startswith("0") and len(key) != 4:
                # key has leading zeros
                raise ValueError("leading zeros in key")
        key = key.replace("-", "_")
    return (ParamKey.make(key), force_generic)


def key_to_text(key):
    return ParamKey.to_text(key).replace("_", "-").lower()


# Like rdata escapify, but escapes ',' too.

_escaped = b'",\\'


def _escapify(qstring):
    text = ""
    for c in qstring:
        if c in _escaped:
            text += "\\" + chr(c)
        elif c >= 0x20 and c < 0x7F:
            text += chr(c)
        else:
            text += "\\%03d" % c
    return text


def _unescape(value):
    if value == "":
        return value
    unescaped = b""
    l = len(value)
    i = 0
    while i < l:
        c = value[i]
        i += 1
        if c == "\\":
            if i >= l:  # pragma: no cover   (can't happen via tokenizer get())
                raise dns.exception.UnexpectedEnd
            c = value[i]
            i += 1
            if c.isdigit():
                if i >= l:
                    raise dns.exception.UnexpectedEnd
                c2 = value[i]
                i += 1
                if i >= l:
                    raise dns.exception.UnexpectedEnd
                c3 = value[i]
                i += 1
                if not (c2.isdigit() and c3.isdigit()):
                    raise dns.exception.SyntaxError
                codepoint = int(c) * 100 + int(c2) * 10 + int(c3)
                if codepoint > 255:
                    raise dns.exception.SyntaxError
                unescaped += b"%c" % (codepoint)
                continue
        unescaped += c.encode()
    return unescaped


def _split(value):
    l = len(value)
    i = 0
    items = []
    unescaped = b""
    while i < l:
        c = value[i]
        i += 1
        if c == ord("\\"):
            if i >= l:  # pragma: no cover   (can't happen via tokenizer get())
                raise dns.exception.UnexpectedEnd
            c = value[i]
            i += 1
            unescaped += b"%c" % (c)
        elif c == ord(","):
            items.append(unescaped)
            unescaped = b""
        else:
            unescaped += b"%c" % (c)
    items.append(unescaped)
    return items


@dns.immutable.immutable
class Param:
    """Abstract base class for SVCB parameters"""

    @classmethod
    def emptiness(cls):
        return Emptiness.NEVER


@dns.immutable.immutable
class GenericParam(Param):
    """Generic SVCB parameter"""

    def __init__(self, value):
        self.value = dns.rdata.Rdata._as_bytes(value, True)

    @classmethod
    def emptiness(cls):
        return Emptiness.ALLOWED

    @classmethod
    def from_value(cls, value):
        if value is None or len(value) == 0:
            return None
        else:
            return cls(_unescape(value))

    def to_text(self):
        return '"' + dns.rdata._escapify(self.value) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        value = parser.get_bytes(parser.remaining())
        if len(value) == 0:
            return None
        else:
            return cls(value)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        file.write(self.value)


@dns.immutable.immutable
class MandatoryParam(Param):
    def __init__(self, keys):
        # check for duplicates
        keys = sorted([_validate_key(key)[0] for key in keys])
        prior_k = None
        for k in keys:
            if k == prior_k:
                raise ValueError(f"duplicate key {k:d}")
            prior_k = k
            if k == ParamKey.MANDATORY:
                raise ValueError("listed the mandatory key as mandatory")
        self.keys = tuple(keys)

    @classmethod
    def from_value(cls, value):
        keys = [k.encode() for k in value.split(",")]
        return cls(keys)

    def to_text(self):
        return '"' + ",".join([key_to_text(key) for key in self.keys]) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        keys = []
        last_key = -1
        while parser.remaining() > 0:
            key = parser.get_uint16()
            if key < last_key:
                raise dns.exception.FormError("manadatory keys not ascending")
            last_key = key
            keys.append(key)
        return cls(keys)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        for key in self.keys:
            file.write(struct.pack("!H", key))


@dns.immutable.immutable
class ALPNParam(Param):
    def __init__(self, ids):
        self.ids = dns.rdata.Rdata._as_tuple(
            ids, lambda x: dns.rdata.Rdata._as_bytes(x, True, 255, False)
        )

    @classmethod
    def from_value(cls, value):
        return cls(_split(_unescape(value)))

    def to_text(self):
        value = ",".join([_escapify(id) for id in self.ids])
        return '"' + dns.rdata._escapify(value.encode()) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        ids = []
        while parser.remaining() > 0:
            id = parser.get_counted_bytes()
            ids.append(id)
        return cls(ids)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        for id in self.ids:
            file.write(struct.pack("!B", len(id)))
            file.write(id)


@dns.immutable.immutable
class NoDefaultALPNParam(Param):
    # We don't ever expect to instantiate this class, but we need
    # a from_value() and a from_wire_parser(), so we just return None
    # from the class methods when things are OK.

    @classmethod
    def emptiness(cls):
        return Emptiness.ALWAYS

    @classmethod
    def from_value(cls, value):
        if value is None or value == "":
            return None
        else:
            raise ValueError("no-default-alpn with non-empty value")

    def to_text(self):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        if parser.remaining() != 0:
            raise dns.exception.FormError
        return None

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        raise NotImplementedError  # pragma: no cover


@dns.immutable.immutable
class PortParam(Param):
    def __init__(self, port):
        self.port = dns.rdata.Rdata._as_uint16(port)

    @classmethod
    def from_value(cls, value):
        value = int(value)
        return cls(value)

    def to_text(self):
        return f'"{self.port}"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        port = parser.get_uint16()
        return cls(port)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        file.write(struct.pack("!H", self.port))


@dns.immutable.immutable
class IPv4HintParam(Param):
    def __init__(self, addresses):
        self.addresses = dns.rdata.Rdata._as_tuple(
            addresses, dns.rdata.Rdata._as_ipv4_address
        )

    @classmethod
    def from_value(cls, value):
        addresses = value.split(",")
        return cls(addresses)

    def to_text(self):
        return '"' + ",".join(self.addresses) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        addresses = []
        while parser.remaining() > 0:
            ip = parser.get_bytes(4)
            addresses.append(dns.ipv4.inet_ntoa(ip))
        return cls(addresses)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        for address in self.addresses:
            file.write(dns.ipv4.inet_aton(address))


@dns.immutable.immutable
class IPv6HintParam(Param):
    def __init__(self, addresses):
        self.addresses = dns.rdata.Rdata._as_tuple(
            addresses, dns.rdata.Rdata._as_ipv6_address
        )

    @classmethod
    def from_value(cls, value):
        addresses = value.split(",")
        return cls(addresses)

    def to_text(self):
        return '"' + ",".join(self.addresses) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        addresses = []
        while parser.remaining() > 0:
            ip = parser.get_bytes(16)
            addresses.append(dns.ipv6.inet_ntoa(ip))
        return cls(addresses)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        for address in self.addresses:
            file.write(dns.ipv6.inet_aton(address))


@dns.immutable.immutable
class ECHParam(Param):
    def __init__(self, ech):
        self.ech = dns.rdata.Rdata._as_bytes(ech, True)

    @classmethod
    def from_value(cls, value):
        if "\\" in value:
            raise ValueError("escape in ECH value")
        value = base64.b64decode(value.encode())
        return cls(value)

    def to_text(self):
        b64 = base64.b64encode(self.ech).decode("ascii")
        return f'"{b64}"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):  # pylint: disable=W0613
        value = parser.get_bytes(parser.remaining())
        return cls(value)

    def to_wire(self, file, origin=None):  # pylint: disable=W0613
        file.write(self.ech)


_class_for_key = {
    ParamKey.MANDATORY: MandatoryParam,
    ParamKey.ALPN: ALPNParam,
    ParamKey.NO_DEFAULT_ALPN: NoDefaultALPNParam,
    ParamKey.PORT: PortParam,
    ParamKey.IPV4HINT: IPv4HintParam,
    ParamKey.ECH: ECHParam,
    ParamKey.IPV6HINT: IPv6HintParam,
}


def _validate_and_define(params, key, value):
    (key, force_generic) = _validate_key(_unescape(key))
    if key in params:
        raise SyntaxError(f'duplicate key "{key:d}"')
    cls = _class_for_key.get(key, GenericParam)
    emptiness = cls.emptiness()
    if value is None:
        if emptiness == Emptiness.NEVER:
            raise SyntaxError("value cannot be empty")
        value = cls.from_value(value)
    else:
        if force_generic:
            value = cls.from_wire_parser(dns.wire.Parser(_unescape(value)))
        else:
            value = cls.from_value(value)
    params[key] = value


@dns.immutable.immutable
class SVCBBase(dns.rdata.Rdata):

    """Base class for SVCB-like records"""

    # see: draft-ietf-dnsop-svcb-https-11

    __slots__ = ["priority", "target", "params"]

    def __init__(self, rdclass, rdtype, priority, target, params):
        super().__init__(rdclass, rdtype)
        self.priority = self._as_uint16(priority)
        self.target = self._as_name(target)
        for k, v in params.items():
            k = ParamKey.make(k)
            if not isinstance(v, Param) and v is not None:
                raise ValueError(f"{k:d} not a Param")
        self.params = dns.immutable.Dict(params)
        # Make sure any parameter listed as mandatory is present in the
        # record.
        mandatory = params.get(ParamKey.MANDATORY)
        if mandatory:
            for key in mandatory.keys:
                # Note we have to say "not in" as we have None as a value
                # so a get() and a not None test would be wrong.
                if key not in params:
                    raise ValueError(f"key {key:d} declared mandatory but not present")
        # The no-default-alpn parameter requires the alpn parameter.
        if ParamKey.NO_DEFAULT_ALPN in params:
            if ParamKey.ALPN not in params:
                raise ValueError("no-default-alpn present, but alpn missing")

    def to_text(self, origin=None, relativize=True, **kw):
        target = self.target.choose_relativity(origin, relativize)
        params = []
        for key in sorted(self.params.keys()):
            value = self.params[key]
            if value is None:
                params.append(key_to_text(key))
            else:
                kv = key_to_text(key) + "=" + value.to_text()
                params.append(kv)
        if len(params) > 0:
            space = " "
        else:
            space = ""
        return "%d %s%s%s" % (self.priority, target, space, " ".join(params))

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        priority = tok.get_uint16()
        target = tok.get_name(origin, relativize, relativize_to)
        if priority == 0:
            token = tok.get()
            if not token.is_eol_or_eof():
                raise SyntaxError("parameters in AliasMode")
            tok.unget(token)
        params = {}
        while True:
            token = tok.get()
            if token.is_eol_or_eof():
                tok.unget(token)
                break
            if token.ttype != dns.tokenizer.IDENTIFIER:
                raise SyntaxError("parameter is not an identifier")
            equals = token.value.find("=")
            if equals == len(token.value) - 1:
                # 'key=', so next token should be a quoted string without
                # any intervening whitespace.
                key = token.value[:-1]
                token = tok.get(want_leading=True)
                if token.ttype != dns.tokenizer.QUOTED_STRING:
                    raise SyntaxError("whitespace after =")
                value = token.value
            elif equals > 0:
                # key=value
                key = token.value[:equals]
                value = token.value[equals + 1 :]
            elif equals == 0:
                # =key
                raise SyntaxError('parameter cannot start with "="')
            else:
                # key
                key = token.value
                value = None
            _validate_and_define(params, key, value)
        return cls(rdclass, rdtype, priority, target, params)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(struct.pack("!H", self.priority))
        self.target.to_wire(file, None, origin, False)
        for key in sorted(self.params):
            file.write(struct.pack("!H", key))
            value = self.params[key]
            # placeholder for length (or actual length of empty values)
            file.write(struct.pack("!H", 0))
            if value is None:
                continue
            else:
                start = file.tell()
                value.to_wire(file, origin)
                end = file.tell()
                assert end - start < 65536
                file.seek(start - 2)
                stuff = struct.pack("!H", end - start)
                file.write(stuff)
                file.seek(0, io.SEEK_END)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        priority = parser.get_uint16()
        target = parser.get_name(origin)
        if priority == 0 and parser.remaining() != 0:
            raise dns.exception.FormError("parameters in AliasMode")
        params = {}
        prior_key = -1
        while parser.remaining() > 0:
            key = parser.get_uint16()
            if key < prior_key:
                raise dns.exception.FormError("keys not in order")
            prior_key = key
            vlen = parser.get_uint16()
            pcls = _class_for_key.get(key, GenericParam)
            with parser.restrict_to(vlen):
                value = pcls.from_wire_parser(parser, origin)
            params[key] = value
        return cls(rdclass, rdtype, priority, target, params)

    def _processing_priority(self):
        return self.priority

    @classmethod
    def _processing_order(cls, iterable):
        return dns.rdtypes.util.priority_processing_order(iterable)
