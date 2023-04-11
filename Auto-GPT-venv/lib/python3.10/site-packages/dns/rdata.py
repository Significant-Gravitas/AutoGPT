# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2001-2017 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""DNS rdata."""

from typing import Any, Dict, Optional, Tuple, Union

from importlib import import_module
import base64
import binascii
import io
import inspect
import itertools
import random

import dns.wire
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.ttl

_chunksize = 32

# We currently allow comparisons for rdata with relative names for backwards
# compatibility, but in the future we will not, as these kinds of comparisons
# can lead to subtle bugs if code is not carefully written.
#
# This switch allows the future behavior to be turned on so code can be
# tested with it.
_allow_relative_comparisons = True


class NoRelativeRdataOrdering(dns.exception.DNSException):
    """An attempt was made to do an ordered comparison of one or more
    rdata with relative names.  The only reliable way of sorting rdata
    is to use non-relativized rdata.

    """


def _wordbreak(data, chunksize=_chunksize, separator=b" "):
    """Break a binary string into chunks of chunksize characters separated by
    a space.
    """

    if not chunksize:
        return data.decode()
    return separator.join(
        [data[i : i + chunksize] for i in range(0, len(data), chunksize)]
    ).decode()


# pylint: disable=unused-argument


def _hexify(data, chunksize=_chunksize, separator=b" ", **kw):
    """Convert a binary string into its hex encoding, broken up into chunks
    of chunksize characters separated by a separator.
    """

    return _wordbreak(binascii.hexlify(data), chunksize, separator)


def _base64ify(data, chunksize=_chunksize, separator=b" ", **kw):
    """Convert a binary string into its base64 encoding, broken up into chunks
    of chunksize characters separated by a separator.
    """

    return _wordbreak(base64.b64encode(data), chunksize, separator)


# pylint: enable=unused-argument

__escaped = b'"\\'


def _escapify(qstring):
    """Escape the characters in a quoted string which need it."""

    if isinstance(qstring, str):
        qstring = qstring.encode()
    if not isinstance(qstring, bytearray):
        qstring = bytearray(qstring)

    text = ""
    for c in qstring:
        if c in __escaped:
            text += "\\" + chr(c)
        elif c >= 0x20 and c < 0x7F:
            text += chr(c)
        else:
            text += "\\%03d" % c
    return text


def _truncate_bitmap(what):
    """Determine the index of greatest byte that isn't all zeros, and
    return the bitmap that contains all the bytes less than that index.
    """

    for i in range(len(what) - 1, -1, -1):
        if what[i] != 0:
            return what[0 : i + 1]
    return what[0:1]


# So we don't have to edit all the rdata classes...
_constify = dns.immutable.constify


@dns.immutable.immutable
class Rdata:
    """Base class for all DNS rdata types."""

    __slots__ = ["rdclass", "rdtype", "rdcomment"]

    def __init__(self, rdclass, rdtype):
        """Initialize an rdata.

        *rdclass*, an ``int`` is the rdataclass of the Rdata.

        *rdtype*, an ``int`` is the rdatatype of the Rdata.
        """

        self.rdclass = self._as_rdataclass(rdclass)
        self.rdtype = self._as_rdatatype(rdtype)
        self.rdcomment = None

    def _get_all_slots(self):
        return itertools.chain.from_iterable(
            getattr(cls, "__slots__", []) for cls in self.__class__.__mro__
        )

    def __getstate__(self):
        # We used to try to do a tuple of all slots here, but it
        # doesn't work as self._all_slots isn't available at
        # __setstate__() time.  Before that we tried to store a tuple
        # of __slots__, but that didn't work as it didn't store the
        # slots defined by ancestors.  This older way didn't fail
        # outright, but ended up with partially broken objects, e.g.
        # if you unpickled an A RR it wouldn't have rdclass and rdtype
        # attributes, and would compare badly.
        state = {}
        for slot in self._get_all_slots():
            state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state):
        for slot, val in state.items():
            object.__setattr__(self, slot, val)
        if not hasattr(self, "rdcomment"):
            # Pickled rdata from 2.0.x might not have a rdcomment, so add
            # it if needed.
            object.__setattr__(self, "rdcomment", None)

    def covers(self) -> dns.rdatatype.RdataType:
        """Return the type a Rdata covers.

        DNS SIG/RRSIG rdatas apply to a specific type; this type is
        returned by the covers() function.  If the rdata type is not
        SIG or RRSIG, dns.rdatatype.NONE is returned.  This is useful when
        creating rdatasets, allowing the rdataset to contain only RRSIGs
        of a particular type, e.g. RRSIG(NS).

        Returns a ``dns.rdatatype.RdataType``.
        """

        return dns.rdatatype.NONE

    def extended_rdatatype(self) -> int:
        """Return a 32-bit type value, the least significant 16 bits of
        which are the ordinary DNS type, and the upper 16 bits of which are
        the "covered" type, if any.

        Returns an ``int``.
        """

        return self.covers() << 16 | self.rdtype

    def to_text(
        self,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        **kw: Dict[str, Any]
    ) -> str:
        """Convert an rdata to text format.

        Returns a ``str``.
        """

        raise NotImplementedError  # pragma: no cover

    def _to_wire(
        self,
        file: Optional[Any],
        compress: Optional[dns.name.CompressType] = None,
        origin: Optional[dns.name.Name] = None,
        canonicalize: bool = False,
    ) -> bytes:
        raise NotImplementedError  # pragma: no cover

    def to_wire(
        self,
        file: Optional[Any] = None,
        compress: Optional[dns.name.CompressType] = None,
        origin: Optional[dns.name.Name] = None,
        canonicalize: bool = False,
    ) -> bytes:
        """Convert an rdata to wire format.

        Returns a ``bytes`` or ``None``.
        """

        if file:
            return self._to_wire(file, compress, origin, canonicalize)
        else:
            f = io.BytesIO()
            self._to_wire(f, compress, origin, canonicalize)
            return f.getvalue()

    def to_generic(
        self, origin: Optional[dns.name.Name] = None
    ) -> "dns.rdata.GenericRdata":
        """Creates a dns.rdata.GenericRdata equivalent of this rdata.

        Returns a ``dns.rdata.GenericRdata``.
        """
        return dns.rdata.GenericRdata(
            self.rdclass, self.rdtype, self.to_wire(origin=origin)
        )

    def to_digestable(self, origin: Optional[dns.name.Name] = None) -> bytes:
        """Convert rdata to a format suitable for digesting in hashes.  This
        is also the DNSSEC canonical form.

        Returns a ``bytes``.
        """

        return self.to_wire(origin=origin, canonicalize=True)

    def __repr__(self):
        covers = self.covers()
        if covers == dns.rdatatype.NONE:
            ctext = ""
        else:
            ctext = "(" + dns.rdatatype.to_text(covers) + ")"
        return (
            "<DNS "
            + dns.rdataclass.to_text(self.rdclass)
            + " "
            + dns.rdatatype.to_text(self.rdtype)
            + ctext
            + " rdata: "
            + str(self)
            + ">"
        )

    def __str__(self):
        return self.to_text()

    def _cmp(self, other):
        """Compare an rdata with another rdata of the same rdtype and
        rdclass.

        For rdata with only absolute names:
            Return < 0 if self < other in the DNSSEC ordering, 0 if self
            == other, and > 0 if self > other.
        For rdata with at least one relative names:
            The rdata sorts before any rdata with only absolute names.
            When compared with another relative rdata, all names are
            made absolute as if they were relative to the root, as the
            proper origin is not available.  While this creates a stable
            ordering, it is NOT guaranteed to be the DNSSEC ordering.
            In the future, all ordering comparisons for rdata with
            relative names will be disallowed.
        """
        try:
            our = self.to_digestable()
            our_relative = False
        except dns.name.NeedAbsoluteNameOrOrigin:
            if _allow_relative_comparisons:
                our = self.to_digestable(dns.name.root)
            our_relative = True
        try:
            their = other.to_digestable()
            their_relative = False
        except dns.name.NeedAbsoluteNameOrOrigin:
            if _allow_relative_comparisons:
                their = other.to_digestable(dns.name.root)
            their_relative = True
        if _allow_relative_comparisons:
            if our_relative != their_relative:
                # For the purpose of comparison, all rdata with at least one
                # relative name is less than an rdata with only absolute names.
                if our_relative:
                    return -1
                else:
                    return 1
        elif our_relative or their_relative:
            raise NoRelativeRdataOrdering
        if our == their:
            return 0
        elif our > their:
            return 1
        else:
            return -1

    def __eq__(self, other):
        if not isinstance(other, Rdata):
            return False
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return False
        our_relative = False
        their_relative = False
        try:
            our = self.to_digestable()
        except dns.name.NeedAbsoluteNameOrOrigin:
            our = self.to_digestable(dns.name.root)
            our_relative = True
        try:
            their = other.to_digestable()
        except dns.name.NeedAbsoluteNameOrOrigin:
            their = other.to_digestable(dns.name.root)
            their_relative = True
        if our_relative != their_relative:
            return False
        return our == their

    def __ne__(self, other):
        if not isinstance(other, Rdata):
            return True
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return True
        return not self.__eq__(other)

    def __lt__(self, other):
        if (
            not isinstance(other, Rdata)
            or self.rdclass != other.rdclass
            or self.rdtype != other.rdtype
        ):

            return NotImplemented
        return self._cmp(other) < 0

    def __le__(self, other):
        if (
            not isinstance(other, Rdata)
            or self.rdclass != other.rdclass
            or self.rdtype != other.rdtype
        ):
            return NotImplemented
        return self._cmp(other) <= 0

    def __ge__(self, other):
        if (
            not isinstance(other, Rdata)
            or self.rdclass != other.rdclass
            or self.rdtype != other.rdtype
        ):
            return NotImplemented
        return self._cmp(other) >= 0

    def __gt__(self, other):
        if (
            not isinstance(other, Rdata)
            or self.rdclass != other.rdclass
            or self.rdtype != other.rdtype
        ):
            return NotImplemented
        return self._cmp(other) > 0

    def __hash__(self):
        return hash(self.to_digestable(dns.name.root))

    @classmethod
    def from_text(
        cls,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        tok: dns.tokenizer.Tokenizer,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        relativize_to: Optional[dns.name.Name] = None,
    ) -> "Rdata":
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_wire_parser(
        cls,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        parser: dns.wire.Parser,
        origin: Optional[dns.name.Name] = None,
    ) -> "Rdata":
        raise NotImplementedError  # pragma: no cover

    def replace(self, **kwargs: Any) -> "Rdata":
        """
        Create a new Rdata instance based on the instance replace was
        invoked on. It is possible to pass different parameters to
        override the corresponding properties of the base Rdata.

        Any field specific to the Rdata type can be replaced, but the
        *rdtype* and *rdclass* fields cannot.

        Returns an instance of the same Rdata subclass as *self*.
        """

        # Get the constructor parameters.
        parameters = inspect.signature(self.__init__).parameters  # type: ignore

        # Ensure that all of the arguments correspond to valid fields.
        # Don't allow rdclass or rdtype to be changed, though.
        for key in kwargs:
            if key == "rdcomment":
                continue
            if key not in parameters:
                raise AttributeError(
                    "'{}' object has no attribute '{}'".format(
                        self.__class__.__name__, key
                    )
                )
            if key in ("rdclass", "rdtype"):
                raise AttributeError(
                    "Cannot overwrite '{}' attribute '{}'".format(
                        self.__class__.__name__, key
                    )
                )

        # Construct the parameter list.  For each field, use the value in
        # kwargs if present, and the current value otherwise.
        args = (kwargs.get(key, getattr(self, key)) for key in parameters)

        # Create, validate, and return the new object.
        rd = self.__class__(*args)
        # The comment is not set in the constructor, so give it special
        # handling.
        rdcomment = kwargs.get("rdcomment", self.rdcomment)
        if rdcomment is not None:
            object.__setattr__(rd, "rdcomment", rdcomment)
        return rd

    # Type checking and conversion helpers.  These are class methods as
    # they don't touch object state and may be useful to others.

    @classmethod
    def _as_rdataclass(cls, value):
        return dns.rdataclass.RdataClass.make(value)

    @classmethod
    def _as_rdatatype(cls, value):
        return dns.rdatatype.RdataType.make(value)

    @classmethod
    def _as_bytes(
        cls,
        value: Any,
        encode: bool = False,
        max_length: Optional[int] = None,
        empty_ok: bool = True,
    ) -> bytes:
        if encode and isinstance(value, str):
            bvalue = value.encode()
        elif isinstance(value, bytearray):
            bvalue = bytes(value)
        elif isinstance(value, bytes):
            bvalue = value
        else:
            raise ValueError("not bytes")
        if max_length is not None and len(bvalue) > max_length:
            raise ValueError("too long")
        if not empty_ok and len(bvalue) == 0:
            raise ValueError("empty bytes not allowed")
        return bvalue

    @classmethod
    def _as_name(cls, value):
        # Note that proper name conversion (e.g. with origin and IDNA
        # awareness) is expected to be done via from_text.  This is just
        # a simple thing for people invoking the constructor directly.
        if isinstance(value, str):
            return dns.name.from_text(value)
        elif not isinstance(value, dns.name.Name):
            raise ValueError("not a name")
        return value

    @classmethod
    def _as_uint8(cls, value):
        if not isinstance(value, int):
            raise ValueError("not an integer")
        if value < 0 or value > 255:
            raise ValueError("not a uint8")
        return value

    @classmethod
    def _as_uint16(cls, value):
        if not isinstance(value, int):
            raise ValueError("not an integer")
        if value < 0 or value > 65535:
            raise ValueError("not a uint16")
        return value

    @classmethod
    def _as_uint32(cls, value):
        if not isinstance(value, int):
            raise ValueError("not an integer")
        if value < 0 or value > 4294967295:
            raise ValueError("not a uint32")
        return value

    @classmethod
    def _as_uint48(cls, value):
        if not isinstance(value, int):
            raise ValueError("not an integer")
        if value < 0 or value > 281474976710655:
            raise ValueError("not a uint48")
        return value

    @classmethod
    def _as_int(cls, value, low=None, high=None):
        if not isinstance(value, int):
            raise ValueError("not an integer")
        if low is not None and value < low:
            raise ValueError("value too small")
        if high is not None and value > high:
            raise ValueError("value too large")
        return value

    @classmethod
    def _as_ipv4_address(cls, value):
        if isinstance(value, str):
            # call to check validity
            dns.ipv4.inet_aton(value)
            return value
        elif isinstance(value, bytes):
            return dns.ipv4.inet_ntoa(value)
        else:
            raise ValueError("not an IPv4 address")

    @classmethod
    def _as_ipv6_address(cls, value):
        if isinstance(value, str):
            # call to check validity
            dns.ipv6.inet_aton(value)
            return value
        elif isinstance(value, bytes):
            return dns.ipv6.inet_ntoa(value)
        else:
            raise ValueError("not an IPv6 address")

    @classmethod
    def _as_bool(cls, value):
        if isinstance(value, bool):
            return value
        else:
            raise ValueError("not a boolean")

    @classmethod
    def _as_ttl(cls, value):
        if isinstance(value, int):
            return cls._as_int(value, 0, dns.ttl.MAX_TTL)
        elif isinstance(value, str):
            return dns.ttl.from_text(value)
        else:
            raise ValueError("not a TTL")

    @classmethod
    def _as_tuple(cls, value, as_value):
        try:
            # For user convenience, if value is a singleton of the list
            # element type, wrap it in a tuple.
            return (as_value(value),)
        except Exception:
            # Otherwise, check each element of the iterable *value*
            # against *as_value*.
            return tuple(as_value(v) for v in value)

    # Processing order

    @classmethod
    def _processing_order(cls, iterable):
        items = list(iterable)
        random.shuffle(items)
        return items


@dns.immutable.immutable
class GenericRdata(Rdata):

    """Generic Rdata Class

    This class is used for rdata types for which we have no better
    implementation.  It implements the DNS "unknown RRs" scheme.
    """

    __slots__ = ["data"]

    def __init__(self, rdclass, rdtype, data):
        super().__init__(rdclass, rdtype)
        self.data = data

    def to_text(
        self,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        **kw: Dict[str, Any]
    ) -> str:
        return r"\# %d " % len(self.data) + _hexify(self.data, **kw)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        token = tok.get()
        if not token.is_identifier() or token.value != r"\#":
            raise dns.exception.SyntaxError(r"generic rdata does not start with \#")
        length = tok.get_int()
        hex = tok.concatenate_remaining_identifiers(True).encode()
        data = binascii.unhexlify(hex)
        if len(data) != length:
            raise dns.exception.SyntaxError("generic rdata hex data has wrong length")
        return cls(rdclass, rdtype, data)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(self.data)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        return cls(rdclass, rdtype, parser.get_remaining())


_rdata_classes: Dict[
    Tuple[dns.rdataclass.RdataClass, dns.rdatatype.RdataType], Any
] = {}
_module_prefix = "dns.rdtypes"


def get_rdata_class(rdclass, rdtype):
    cls = _rdata_classes.get((rdclass, rdtype))
    if not cls:
        cls = _rdata_classes.get((dns.rdatatype.ANY, rdtype))
        if not cls:
            rdclass_text = dns.rdataclass.to_text(rdclass)
            rdtype_text = dns.rdatatype.to_text(rdtype)
            rdtype_text = rdtype_text.replace("-", "_")
            try:
                mod = import_module(
                    ".".join([_module_prefix, rdclass_text, rdtype_text])
                )
                cls = getattr(mod, rdtype_text)
                _rdata_classes[(rdclass, rdtype)] = cls
            except ImportError:
                try:
                    mod = import_module(".".join([_module_prefix, "ANY", rdtype_text]))
                    cls = getattr(mod, rdtype_text)
                    _rdata_classes[(dns.rdataclass.ANY, rdtype)] = cls
                    _rdata_classes[(rdclass, rdtype)] = cls
                except ImportError:
                    pass
    if not cls:
        cls = GenericRdata
        _rdata_classes[(rdclass, rdtype)] = cls
    return cls


def from_text(
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    tok: Union[dns.tokenizer.Tokenizer, str],
    origin: Optional[dns.name.Name] = None,
    relativize: bool = True,
    relativize_to: Optional[dns.name.Name] = None,
    idna_codec: Optional[dns.name.IDNACodec] = None,
) -> Rdata:
    """Build an rdata object from text format.

    This function attempts to dynamically load a class which
    implements the specified rdata class and type.  If there is no
    class-and-type-specific implementation, the GenericRdata class
    is used.

    Once a class is chosen, its from_text() class method is called
    with the parameters to this function.

    If *tok* is a ``str``, then a tokenizer is created and the string
    is used as its input.

    *rdclass*, a ``dns.rdataclass.RdataClass`` or ``str``, the rdataclass.

    *rdtype*, a ``dns.rdatatype.RdataType`` or ``str``, the rdatatype.

    *tok*, a ``dns.tokenizer.Tokenizer`` or a ``str``.

    *origin*, a ``dns.name.Name`` (or ``None``), the
    origin to use for relative names.

    *relativize*, a ``bool``.  If true, name will be relativized.

    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use
    when relativizing names.  If not set, the *origin* value will be used.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder to use if a tokenizer needs to be created.  If
    ``None``, the default IDNA 2003 encoder/decoder is used.  If a
    tokenizer is not created, then the codec associated with the tokenizer
    is the one that is used.

    Returns an instance of the chosen Rdata subclass.

    """
    if isinstance(tok, str):
        tok = dns.tokenizer.Tokenizer(tok, idna_codec=idna_codec)
    rdclass = dns.rdataclass.RdataClass.make(rdclass)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    cls = get_rdata_class(rdclass, rdtype)
    with dns.exception.ExceptionWrapper(dns.exception.SyntaxError):
        rdata = None
        if cls != GenericRdata:
            # peek at first token
            token = tok.get()
            tok.unget(token)
            if token.is_identifier() and token.value == r"\#":
                #
                # Known type using the generic syntax.  Extract the
                # wire form from the generic syntax, and then run
                # from_wire on it.
                #
                grdata = GenericRdata.from_text(
                    rdclass, rdtype, tok, origin, relativize, relativize_to
                )
                rdata = from_wire(
                    rdclass, rdtype, grdata.data, 0, len(grdata.data), origin
                )
                #
                # If this comparison isn't equal, then there must have been
                # compressed names in the wire format, which is an error,
                # there being no reasonable context to decompress with.
                #
                rwire = rdata.to_wire()
                if rwire != grdata.data:
                    raise dns.exception.SyntaxError(
                        "compressed data in "
                        "generic syntax form "
                        "of known rdatatype"
                    )
        if rdata is None:
            rdata = cls.from_text(
                rdclass, rdtype, tok, origin, relativize, relativize_to
            )
        token = tok.get_eol_as_token()
        if token.comment is not None:
            object.__setattr__(rdata, "rdcomment", token.comment)
        return rdata


def from_wire_parser(
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    parser: dns.wire.Parser,
    origin: Optional[dns.name.Name] = None,
) -> Rdata:
    """Build an rdata object from wire format

    This function attempts to dynamically load a class which
    implements the specified rdata class and type.  If there is no
    class-and-type-specific implementation, the GenericRdata class
    is used.

    Once a class is chosen, its from_wire() class method is called
    with the parameters to this function.

    *rdclass*, a ``dns.rdataclass.RdataClass`` or ``str``, the rdataclass.

    *rdtype*, a ``dns.rdatatype.RdataType`` or ``str``, the rdatatype.

    *parser*, a ``dns.wire.Parser``, the parser, which should be
    restricted to the rdata length.

    *origin*, a ``dns.name.Name`` (or ``None``).  If not ``None``,
    then names will be relativized to this origin.

    Returns an instance of the chosen Rdata subclass.
    """

    rdclass = dns.rdataclass.RdataClass.make(rdclass)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    cls = get_rdata_class(rdclass, rdtype)
    with dns.exception.ExceptionWrapper(dns.exception.FormError):
        return cls.from_wire_parser(rdclass, rdtype, parser, origin)


def from_wire(
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    wire: bytes,
    current: int,
    rdlen: int,
    origin: Optional[dns.name.Name] = None,
) -> Rdata:
    """Build an rdata object from wire format

    This function attempts to dynamically load a class which
    implements the specified rdata class and type.  If there is no
    class-and-type-specific implementation, the GenericRdata class
    is used.

    Once a class is chosen, its from_wire() class method is called
    with the parameters to this function.

    *rdclass*, an ``int``, the rdataclass.

    *rdtype*, an ``int``, the rdatatype.

    *wire*, a ``bytes``, the wire-format message.

    *current*, an ``int``, the offset in wire of the beginning of
    the rdata.

    *rdlen*, an ``int``, the length of the wire-format rdata

    *origin*, a ``dns.name.Name`` (or ``None``).  If not ``None``,
    then names will be relativized to this origin.

    Returns an instance of the chosen Rdata subclass.
    """
    parser = dns.wire.Parser(wire, current)
    with parser.restrict_to(rdlen):
        return from_wire_parser(rdclass, rdtype, parser, origin)


class RdatatypeExists(dns.exception.DNSException):
    """DNS rdatatype already exists."""

    supp_kwargs = {"rdclass", "rdtype"}
    fmt = (
        "The rdata type with class {rdclass:d} and rdtype {rdtype:d} "
        + "already exists."
    )


def register_type(
    implementation: Any,
    rdtype: int,
    rdtype_text: str,
    is_singleton: bool = False,
    rdclass: dns.rdataclass.RdataClass = dns.rdataclass.IN,
) -> None:
    """Dynamically register a module to handle an rdatatype.

    *implementation*, a module implementing the type in the usual dnspython
    way.

    *rdtype*, an ``int``, the rdatatype to register.

    *rdtype_text*, a ``str``, the textual form of the rdatatype.

    *is_singleton*, a ``bool``, indicating if the type is a singleton (i.e.
    RRsets of the type can have only one member.)

    *rdclass*, the rdataclass of the type, or ``dns.rdataclass.ANY`` if
    it applies to all classes.
    """

    the_rdtype = dns.rdatatype.RdataType.make(rdtype)
    existing_cls = get_rdata_class(rdclass, the_rdtype)
    if existing_cls != GenericRdata or dns.rdatatype.is_metatype(the_rdtype):
        raise RdatatypeExists(rdclass=rdclass, rdtype=the_rdtype)
    try:
        if dns.rdatatype.RdataType(the_rdtype).name != rdtype_text:
            raise RdatatypeExists(rdclass=rdclass, rdtype=the_rdtype)
    except ValueError:
        pass
    _rdata_classes[(rdclass, the_rdtype)] = getattr(
        implementation, rdtype_text.replace("-", "_")
    )
    dns.rdatatype.register_type(the_rdtype, rdtype_text, is_singleton)
