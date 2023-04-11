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

"""DNS Names.
"""

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import copy
import struct

import encodings.idna  # type: ignore

try:
    import idna  # type: ignore

    have_idna_2008 = True
except ImportError:  # pragma: no cover
    have_idna_2008 = False

import dns.enum
import dns.wire
import dns.exception
import dns.immutable


CompressType = Dict["Name", int]


class NameRelation(dns.enum.IntEnum):
    """Name relation result from fullcompare()."""

    # This is an IntEnum for backwards compatibility in case anyone
    # has hardwired the constants.

    #: The compared names have no relationship to each other.
    NONE = 0
    #: the first name is a superdomain of the second.
    SUPERDOMAIN = 1
    #: The first name is a subdomain of the second.
    SUBDOMAIN = 2
    #: The compared names are equal.
    EQUAL = 3
    #: The compared names have a common ancestor.
    COMMONANCESTOR = 4

    @classmethod
    def _maximum(cls):
        return cls.COMMONANCESTOR

    @classmethod
    def _short_name(cls):
        return cls.__name__


# Backwards compatibility
NAMERELN_NONE = NameRelation.NONE
NAMERELN_SUPERDOMAIN = NameRelation.SUPERDOMAIN
NAMERELN_SUBDOMAIN = NameRelation.SUBDOMAIN
NAMERELN_EQUAL = NameRelation.EQUAL
NAMERELN_COMMONANCESTOR = NameRelation.COMMONANCESTOR


class EmptyLabel(dns.exception.SyntaxError):
    """A DNS label is empty."""


class BadEscape(dns.exception.SyntaxError):
    """An escaped code in a text format of DNS name is invalid."""


class BadPointer(dns.exception.FormError):
    """A DNS compression pointer points forward instead of backward."""


class BadLabelType(dns.exception.FormError):
    """The label type in DNS name wire format is unknown."""


class NeedAbsoluteNameOrOrigin(dns.exception.DNSException):
    """An attempt was made to convert a non-absolute name to
    wire when there was also a non-absolute (or missing) origin."""


class NameTooLong(dns.exception.FormError):
    """A DNS name is > 255 octets long."""


class LabelTooLong(dns.exception.SyntaxError):
    """A DNS label is > 63 octets long."""


class AbsoluteConcatenation(dns.exception.DNSException):
    """An attempt was made to append anything other than the
    empty name to an absolute DNS name."""


class NoParent(dns.exception.DNSException):
    """An attempt was made to get the parent of the root name
    or the empty name."""


class NoIDNA2008(dns.exception.DNSException):
    """IDNA 2008 processing was requested but the idna module is not
    available."""


class IDNAException(dns.exception.DNSException):
    """IDNA processing raised an exception."""

    supp_kwargs = {"idna_exception"}
    fmt = "IDNA processing exception: {idna_exception}"

    # We do this as otherwise mypy complains about unexpected keyword argument
    # idna_exception
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


_escaped = b'"().;\\@$'
_escaped_text = '"().;\\@$'


def _escapify(label: Union[bytes, str]) -> str:
    """Escape the characters in label which need it.
    @returns: the escaped string
    @rtype: string"""
    if isinstance(label, bytes):
        # Ordinary DNS label mode.  Escape special characters and values
        # < 0x20 or > 0x7f.
        text = ""
        for c in label:
            if c in _escaped:
                text += "\\" + chr(c)
            elif c > 0x20 and c < 0x7F:
                text += chr(c)
            else:
                text += "\\%03d" % c
        return text

    # Unicode label mode.  Escape only special characters and values < 0x20
    text = ""
    for uc in label:
        if uc in _escaped_text:
            text += "\\" + uc
        elif uc <= "\x20":
            text += "\\%03d" % ord(uc)
        else:
            text += uc
    return text


class IDNACodec:
    """Abstract base class for IDNA encoder/decoders."""

    def __init__(self):
        pass

    def is_idna(self, label: bytes) -> bool:
        return label.lower().startswith(b"xn--")

    def encode(self, label: str) -> bytes:
        raise NotImplementedError  # pragma: no cover

    def decode(self, label: bytes) -> str:
        # We do not apply any IDNA policy on decode.
        if self.is_idna(label):
            try:
                slabel = label[4:].decode("punycode")
                return _escapify(slabel)
            except Exception as e:
                raise IDNAException(idna_exception=e)
        else:
            return _escapify(label)


class IDNA2003Codec(IDNACodec):
    """IDNA 2003 encoder/decoder."""

    def __init__(self, strict_decode: bool = False):
        """Initialize the IDNA 2003 encoder/decoder.

        *strict_decode* is a ``bool``. If `True`, then IDNA2003 checking
        is done when decoding.  This can cause failures if the name
        was encoded with IDNA2008.  The default is `False`.
        """

        super().__init__()
        self.strict_decode = strict_decode

    def encode(self, label: str) -> bytes:
        """Encode *label*."""

        if label == "":
            return b""
        try:
            return encodings.idna.ToASCII(label)
        except UnicodeError:
            raise LabelTooLong

    def decode(self, label: bytes) -> str:
        """Decode *label*."""
        if not self.strict_decode:
            return super().decode(label)
        if label == b"":
            return ""
        try:
            return _escapify(encodings.idna.ToUnicode(label))
        except Exception as e:
            raise IDNAException(idna_exception=e)


class IDNA2008Codec(IDNACodec):
    """IDNA 2008 encoder/decoder."""

    def __init__(
        self,
        uts_46: bool = False,
        transitional: bool = False,
        allow_pure_ascii: bool = False,
        strict_decode: bool = False,
    ):
        """Initialize the IDNA 2008 encoder/decoder.

        *uts_46* is a ``bool``.  If True, apply Unicode IDNA
        compatibility processing as described in Unicode Technical
        Standard #46 (https://unicode.org/reports/tr46/).
        If False, do not apply the mapping.  The default is False.

        *transitional* is a ``bool``: If True, use the
        "transitional" mode described in Unicode Technical Standard
        #46.  The default is False.

        *allow_pure_ascii* is a ``bool``.  If True, then a label which
        consists of only ASCII characters is allowed.  This is less
        strict than regular IDNA 2008, but is also necessary for mixed
        names, e.g. a name with starting with "_sip._tcp." and ending
        in an IDN suffix which would otherwise be disallowed.  The
        default is False.

        *strict_decode* is a ``bool``: If True, then IDNA2008 checking
        is done when decoding.  This can cause failures if the name
        was encoded with IDNA2003.  The default is False.
        """
        super().__init__()
        self.uts_46 = uts_46
        self.transitional = transitional
        self.allow_pure_ascii = allow_pure_ascii
        self.strict_decode = strict_decode

    def encode(self, label: str) -> bytes:
        if label == "":
            return b""
        if self.allow_pure_ascii and is_all_ascii(label):
            encoded = label.encode("ascii")
            if len(encoded) > 63:
                raise LabelTooLong
            return encoded
        if not have_idna_2008:
            raise NoIDNA2008
        try:
            if self.uts_46:
                label = idna.uts46_remap(label, False, self.transitional)
            return idna.alabel(label)
        except idna.IDNAError as e:
            if e.args[0] == "Label too long":
                raise LabelTooLong
            else:
                raise IDNAException(idna_exception=e)

    def decode(self, label: bytes) -> str:
        if not self.strict_decode:
            return super().decode(label)
        if label == b"":
            return ""
        if not have_idna_2008:
            raise NoIDNA2008
        try:
            ulabel = idna.ulabel(label)
            if self.uts_46:
                ulabel = idna.uts46_remap(ulabel, False, self.transitional)
            return _escapify(ulabel)
        except (idna.IDNAError, UnicodeError) as e:
            raise IDNAException(idna_exception=e)


IDNA_2003_Practical = IDNA2003Codec(False)
IDNA_2003_Strict = IDNA2003Codec(True)
IDNA_2003 = IDNA_2003_Practical
IDNA_2008_Practical = IDNA2008Codec(True, False, True, False)
IDNA_2008_UTS_46 = IDNA2008Codec(True, False, False, False)
IDNA_2008_Strict = IDNA2008Codec(False, False, False, True)
IDNA_2008_Transitional = IDNA2008Codec(True, True, False, False)
IDNA_2008 = IDNA_2008_Practical


def _validate_labels(labels: Tuple[bytes, ...]) -> None:
    """Check for empty labels in the middle of a label sequence,
    labels that are too long, and for too many labels.

    Raises ``dns.name.NameTooLong`` if the name as a whole is too long.

    Raises ``dns.name.EmptyLabel`` if a label is empty (i.e. the root
    label) and appears in a position other than the end of the label
    sequence

    """

    l = len(labels)
    total = 0
    i = -1
    j = 0
    for label in labels:
        ll = len(label)
        total += ll + 1
        if ll > 63:
            raise LabelTooLong
        if i < 0 and label == b"":
            i = j
        j += 1
    if total > 255:
        raise NameTooLong
    if i >= 0 and i != l - 1:
        raise EmptyLabel


def _maybe_convert_to_binary(label: Union[bytes, str]) -> bytes:
    """If label is ``str``, convert it to ``bytes``.  If it is already
    ``bytes`` just return it.

    """

    if isinstance(label, bytes):
        return label
    if isinstance(label, str):
        return label.encode()
    raise ValueError  # pragma: no cover


@dns.immutable.immutable
class Name:

    """A DNS name.

    The dns.name.Name class represents a DNS name as a tuple of
    labels.  Each label is a ``bytes`` in DNS wire format.  Instances
    of the class are immutable.
    """

    __slots__ = ["labels"]

    def __init__(self, labels: Iterable[Union[bytes, str]]):
        """*labels* is any iterable whose values are ``str`` or ``bytes``."""

        blabels = [_maybe_convert_to_binary(x) for x in labels]
        self.labels = tuple(blabels)
        _validate_labels(self.labels)

    def __copy__(self):
        return Name(self.labels)

    def __deepcopy__(self, memo):
        return Name(copy.deepcopy(self.labels, memo))

    def __getstate__(self):
        # Names can be pickled
        return {"labels": self.labels}

    def __setstate__(self, state):
        super().__setattr__("labels", state["labels"])
        _validate_labels(self.labels)

    def is_absolute(self) -> bool:
        """Is the most significant label of this name the root label?

        Returns a ``bool``.
        """

        return len(self.labels) > 0 and self.labels[-1] == b""

    def is_wild(self) -> bool:
        """Is this name wild?  (I.e. Is the least significant label '*'?)

        Returns a ``bool``.
        """

        return len(self.labels) > 0 and self.labels[0] == b"*"

    def __hash__(self) -> int:
        """Return a case-insensitive hash of the name.

        Returns an ``int``.
        """

        h = 0
        for label in self.labels:
            for c in label.lower():
                h += (h << 3) + c
        return h

    def fullcompare(self, other: "Name") -> Tuple[NameRelation, int, int]:
        """Compare two names, returning a 3-tuple
        ``(relation, order, nlabels)``.

        *relation* describes the relation ship between the names,
        and is one of: ``dns.name.NameRelation.NONE``,
        ``dns.name.NameRelation.SUPERDOMAIN``, ``dns.name.NameRelation.SUBDOMAIN``,
        ``dns.name.NameRelation.EQUAL``, or ``dns.name.NameRelation.COMMONANCESTOR``.

        *order* is < 0 if *self* < *other*, > 0 if *self* > *other*, and ==
        0 if *self* == *other*.  A relative name is always less than an
        absolute name.  If both names have the same relativity, then
        the DNSSEC order relation is used to order them.

        *nlabels* is the number of significant labels that the two names
        have in common.

        Here are some examples.  Names ending in "." are absolute names,
        those not ending in "." are relative names.

        =============  =============  ===========  =====  =======
        self           other          relation     order  nlabels
        =============  =============  ===========  =====  =======
        www.example.   www.example.   equal        0      3
        www.example.   example.       subdomain    > 0    2
        example.       www.example.   superdomain  < 0    2
        example1.com.  example2.com.  common anc.  < 0    2
        example1       example2.      none         < 0    0
        example1.      example2       none         > 0    0
        =============  =============  ===========  =====  =======
        """

        sabs = self.is_absolute()
        oabs = other.is_absolute()
        if sabs != oabs:
            if sabs:
                return (NameRelation.NONE, 1, 0)
            else:
                return (NameRelation.NONE, -1, 0)
        l1 = len(self.labels)
        l2 = len(other.labels)
        ldiff = l1 - l2
        if ldiff < 0:
            l = l1
        else:
            l = l2

        order = 0
        nlabels = 0
        namereln = NameRelation.NONE
        while l > 0:
            l -= 1
            l1 -= 1
            l2 -= 1
            label1 = self.labels[l1].lower()
            label2 = other.labels[l2].lower()
            if label1 < label2:
                order = -1
                if nlabels > 0:
                    namereln = NameRelation.COMMONANCESTOR
                return (namereln, order, nlabels)
            elif label1 > label2:
                order = 1
                if nlabels > 0:
                    namereln = NameRelation.COMMONANCESTOR
                return (namereln, order, nlabels)
            nlabels += 1
        order = ldiff
        if ldiff < 0:
            namereln = NameRelation.SUPERDOMAIN
        elif ldiff > 0:
            namereln = NameRelation.SUBDOMAIN
        else:
            namereln = NameRelation.EQUAL
        return (namereln, order, nlabels)

    def is_subdomain(self, other: "Name") -> bool:
        """Is self a subdomain of other?

        Note that the notion of subdomain includes equality, e.g.
        "dnspython.org" is a subdomain of itself.

        Returns a ``bool``.
        """

        (nr, _, _) = self.fullcompare(other)
        if nr == NameRelation.SUBDOMAIN or nr == NameRelation.EQUAL:
            return True
        return False

    def is_superdomain(self, other: "Name") -> bool:
        """Is self a superdomain of other?

        Note that the notion of superdomain includes equality, e.g.
        "dnspython.org" is a superdomain of itself.

        Returns a ``bool``.
        """

        (nr, _, _) = self.fullcompare(other)
        if nr == NameRelation.SUPERDOMAIN or nr == NameRelation.EQUAL:
            return True
        return False

    def canonicalize(self) -> "Name":
        """Return a name which is equal to the current name, but is in
        DNSSEC canonical form.
        """

        return Name([x.lower() for x in self.labels])

    def __eq__(self, other):
        if isinstance(other, Name):
            return self.fullcompare(other)[1] == 0
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, Name):
            return self.fullcompare(other)[1] != 0
        else:
            return True

    def __lt__(self, other):
        if isinstance(other, Name):
            return self.fullcompare(other)[1] < 0
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, Name):
            return self.fullcompare(other)[1] <= 0
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Name):
            return self.fullcompare(other)[1] >= 0
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Name):
            return self.fullcompare(other)[1] > 0
        else:
            return NotImplemented

    def __repr__(self):
        return "<DNS name " + self.__str__() + ">"

    def __str__(self):
        return self.to_text(False)

    def to_text(self, omit_final_dot: bool = False) -> str:
        """Convert name to DNS text format.

        *omit_final_dot* is a ``bool``.  If True, don't emit the final
        dot (denoting the root label) for absolute names.  The default
        is False.

        Returns a ``str``.
        """

        if len(self.labels) == 0:
            return "@"
        if len(self.labels) == 1 and self.labels[0] == b"":
            return "."
        if omit_final_dot and self.is_absolute():
            l = self.labels[:-1]
        else:
            l = self.labels
        s = ".".join(map(_escapify, l))
        return s

    def to_unicode(
        self, omit_final_dot: bool = False, idna_codec: Optional[IDNACodec] = None
    ) -> str:
        """Convert name to Unicode text format.

        IDN ACE labels are converted to Unicode.

        *omit_final_dot* is a ``bool``.  If True, don't emit the final
        dot (denoting the root label) for absolute names.  The default
        is False.
        *idna_codec* specifies the IDNA encoder/decoder.  If None, the
        dns.name.IDNA_2003_Practical encoder/decoder is used.
        The IDNA_2003_Practical decoder does
        not impose any policy, it just decodes punycode, so if you
        don't want checking for compliance, you can use this decoder
        for IDNA2008 as well.

        Returns a ``str``.
        """

        if len(self.labels) == 0:
            return "@"
        if len(self.labels) == 1 and self.labels[0] == b"":
            return "."
        if omit_final_dot and self.is_absolute():
            l = self.labels[:-1]
        else:
            l = self.labels
        if idna_codec is None:
            idna_codec = IDNA_2003_Practical
        return ".".join([idna_codec.decode(x) for x in l])

    def to_digestable(self, origin: Optional["Name"] = None) -> bytes:
        """Convert name to a format suitable for digesting in hashes.

        The name is canonicalized and converted to uncompressed wire
        format.  All names in wire format are absolute.  If the name
        is a relative name, then an origin must be supplied.

        *origin* is a ``dns.name.Name`` or ``None``.  If the name is
        relative and origin is not ``None``, then origin will be appended
        to the name.

        Raises ``dns.name.NeedAbsoluteNameOrOrigin`` if the name is
        relative and no origin was provided.

        Returns a ``bytes``.
        """

        digest = self.to_wire(origin=origin, canonicalize=True)
        assert digest is not None
        return digest

    def to_wire(
        self,
        file: Optional[Any] = None,
        compress: Optional[CompressType] = None,
        origin: Optional["Name"] = None,
        canonicalize: bool = False,
    ) -> Optional[bytes]:
        """Convert name to wire format, possibly compressing it.

        *file* is the file where the name is emitted (typically an
        io.BytesIO file).  If ``None`` (the default), a ``bytes``
        containing the wire name will be returned.

        *compress*, a ``dict``, is the compression table to use.  If
        ``None`` (the default), names will not be compressed.  Note that
        the compression code assumes that compression offset 0 is the
        start of *file*, and thus compression will not be correct
        if this is not the case.

        *origin* is a ``dns.name.Name`` or ``None``.  If the name is
        relative and origin is not ``None``, then *origin* will be appended
        to it.

        *canonicalize*, a ``bool``, indicates whether the name should
        be canonicalized; that is, converted to a format suitable for
        digesting in hashes.

        Raises ``dns.name.NeedAbsoluteNameOrOrigin`` if the name is
        relative and no origin was provided.

        Returns a ``bytes`` or ``None``.
        """

        if file is None:
            out = bytearray()
            for label in self.labels:
                out.append(len(label))
                if canonicalize:
                    out += label.lower()
                else:
                    out += label
            if not self.is_absolute():
                if origin is None or not origin.is_absolute():
                    raise NeedAbsoluteNameOrOrigin
                for label in origin.labels:
                    out.append(len(label))
                    if canonicalize:
                        out += label.lower()
                    else:
                        out += label
            return bytes(out)

        labels: Iterable[bytes]
        if not self.is_absolute():
            if origin is None or not origin.is_absolute():
                raise NeedAbsoluteNameOrOrigin
            labels = list(self.labels)
            labels.extend(list(origin.labels))
        else:
            labels = self.labels
        i = 0
        for label in labels:
            n = Name(labels[i:])
            i += 1
            if compress is not None:
                pos = compress.get(n)
            else:
                pos = None
            if pos is not None:
                value = 0xC000 + pos
                s = struct.pack("!H", value)
                file.write(s)
                break
            else:
                if compress is not None and len(n) > 1:
                    pos = file.tell()
                    if pos <= 0x3FFF:
                        compress[n] = pos
                l = len(label)
                file.write(struct.pack("!B", l))
                if l > 0:
                    if canonicalize:
                        file.write(label.lower())
                    else:
                        file.write(label)
        return None

    def __len__(self) -> int:
        """The length of the name (in labels).

        Returns an ``int``.
        """

        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]

    def __add__(self, other):
        return self.concatenate(other)

    def __sub__(self, other):
        return self.relativize(other)

    def split(self, depth: int) -> Tuple["Name", "Name"]:
        """Split a name into a prefix and suffix names at the specified depth.

        *depth* is an ``int`` specifying the number of labels in the suffix

        Raises ``ValueError`` if *depth* was not >= 0 and <= the length of the
        name.

        Returns the tuple ``(prefix, suffix)``.
        """

        l = len(self.labels)
        if depth == 0:
            return (self, dns.name.empty)
        elif depth == l:
            return (dns.name.empty, self)
        elif depth < 0 or depth > l:
            raise ValueError("depth must be >= 0 and <= the length of the name")
        return (Name(self[:-depth]), Name(self[-depth:]))

    def concatenate(self, other: "Name") -> "Name":
        """Return a new name which is the concatenation of self and other.

        Raises ``dns.name.AbsoluteConcatenation`` if the name is
        absolute and *other* is not the empty name.

        Returns a ``dns.name.Name``.
        """

        if self.is_absolute() and len(other) > 0:
            raise AbsoluteConcatenation
        labels = list(self.labels)
        labels.extend(list(other.labels))
        return Name(labels)

    def relativize(self, origin: "Name") -> "Name":
        """If the name is a subdomain of *origin*, return a new name which is
        the name relative to origin.  Otherwise return the name.

        For example, relativizing ``www.dnspython.org.`` to origin
        ``dnspython.org.`` returns the name ``www``.  Relativizing ``example.``
        to origin ``dnspython.org.`` returns ``example.``.

        Returns a ``dns.name.Name``.
        """

        if origin is not None and self.is_subdomain(origin):
            return Name(self[: -len(origin)])
        else:
            return self

    def derelativize(self, origin: "Name") -> "Name":
        """If the name is a relative name, return a new name which is the
        concatenation of the name and origin.  Otherwise return the name.

        For example, derelativizing ``www`` to origin ``dnspython.org.``
        returns the name ``www.dnspython.org.``.  Derelativizing ``example.``
        to origin ``dnspython.org.`` returns ``example.``.

        Returns a ``dns.name.Name``.
        """

        if not self.is_absolute():
            return self.concatenate(origin)
        else:
            return self

    def choose_relativity(
        self, origin: Optional["Name"] = None, relativize: bool = True
    ) -> "Name":
        """Return a name with the relativity desired by the caller.

        If *origin* is ``None``, then the name is returned.
        Otherwise, if *relativize* is ``True`` the name is
        relativized, and if *relativize* is ``False`` the name is
        derelativized.

        Returns a ``dns.name.Name``.
        """

        if origin:
            if relativize:
                return self.relativize(origin)
            else:
                return self.derelativize(origin)
        else:
            return self

    def parent(self) -> "Name":
        """Return the parent of the name.

        For example, the parent of ``www.dnspython.org.`` is ``dnspython.org``.

        Raises ``dns.name.NoParent`` if the name is either the root name or the
        empty name, and thus has no parent.

        Returns a ``dns.name.Name``.
        """

        if self == root or self == empty:
            raise NoParent
        return Name(self.labels[1:])


#: The root name, '.'
root = Name([b""])

#: The empty name.
empty = Name([])


def from_unicode(
    text: str, origin: Optional[Name] = root, idna_codec: Optional[IDNACodec] = None
) -> Name:
    """Convert unicode text into a Name object.

    Labels are encoded in IDN ACE form according to rules specified by
    the IDNA codec.

    *text*, a ``str``, is the text to convert into a name.

    *origin*, a ``dns.name.Name``, specifies the origin to
    append to non-absolute names.  The default is the root name.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.

    Returns a ``dns.name.Name``.
    """

    if not isinstance(text, str):
        raise ValueError("input to from_unicode() must be a unicode string")
    if not (origin is None or isinstance(origin, Name)):
        raise ValueError("origin must be a Name or None")
    labels = []
    label = ""
    escaping = False
    edigits = 0
    total = 0
    if idna_codec is None:
        idna_codec = IDNA_2003
    if text == "@":
        text = ""
    if text:
        if text in [".", "\u3002", "\uff0e", "\uff61"]:
            return Name([b""])  # no Unicode "u" on this constant!
        for c in text:
            if escaping:
                if edigits == 0:
                    if c.isdigit():
                        total = int(c)
                        edigits += 1
                    else:
                        label += c
                        escaping = False
                else:
                    if not c.isdigit():
                        raise BadEscape
                    total *= 10
                    total += int(c)
                    edigits += 1
                    if edigits == 3:
                        escaping = False
                        label += chr(total)
            elif c in [".", "\u3002", "\uff0e", "\uff61"]:
                if len(label) == 0:
                    raise EmptyLabel
                labels.append(idna_codec.encode(label))
                label = ""
            elif c == "\\":
                escaping = True
                edigits = 0
                total = 0
            else:
                label += c
        if escaping:
            raise BadEscape
        if len(label) > 0:
            labels.append(idna_codec.encode(label))
        else:
            labels.append(b"")

    if (len(labels) == 0 or labels[-1] != b"") and origin is not None:
        labels.extend(list(origin.labels))
    return Name(labels)


def is_all_ascii(text: str) -> bool:
    for c in text:
        if ord(c) > 0x7F:
            return False
    return True


def from_text(
    text: Union[bytes, str],
    origin: Optional[Name] = root,
    idna_codec: Optional[IDNACodec] = None,
) -> Name:
    """Convert text into a Name object.

    *text*, a ``bytes`` or ``str``, is the text to convert into a name.

    *origin*, a ``dns.name.Name``, specifies the origin to
    append to non-absolute names.  The default is the root name.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.

    Returns a ``dns.name.Name``.
    """

    if isinstance(text, str):
        if not is_all_ascii(text):
            # Some codepoint in the input text is > 127, so IDNA applies.
            return from_unicode(text, origin, idna_codec)
        # The input is all ASCII, so treat this like an ordinary non-IDNA
        # domain name.  Note that "all ASCII" is about the input text,
        # not the codepoints in the domain name.  E.g. if text has value
        #
        # r'\150\151\152\153\154\155\156\157\158\159'
        #
        # then it's still "all ASCII" even though the domain name has
        # codepoints > 127.
        text = text.encode("ascii")
    if not isinstance(text, bytes):
        raise ValueError("input to from_text() must be a string")
    if not (origin is None or isinstance(origin, Name)):
        raise ValueError("origin must be a Name or None")
    labels = []
    label = b""
    escaping = False
    edigits = 0
    total = 0
    if text == b"@":
        text = b""
    if text:
        if text == b".":
            return Name([b""])
        for c in text:
            byte_ = struct.pack("!B", c)
            if escaping:
                if edigits == 0:
                    if byte_.isdigit():
                        total = int(byte_)
                        edigits += 1
                    else:
                        label += byte_
                        escaping = False
                else:
                    if not byte_.isdigit():
                        raise BadEscape
                    total *= 10
                    total += int(byte_)
                    edigits += 1
                    if edigits == 3:
                        escaping = False
                        label += struct.pack("!B", total)
            elif byte_ == b".":
                if len(label) == 0:
                    raise EmptyLabel
                labels.append(label)
                label = b""
            elif byte_ == b"\\":
                escaping = True
                edigits = 0
                total = 0
            else:
                label += byte_
        if escaping:
            raise BadEscape
        if len(label) > 0:
            labels.append(label)
        else:
            labels.append(b"")
    if (len(labels) == 0 or labels[-1] != b"") and origin is not None:
        labels.extend(list(origin.labels))
    return Name(labels)


# we need 'dns.wire.Parser' quoted as dns.name and dns.wire depend on each other.


def from_wire_parser(parser: "dns.wire.Parser") -> Name:
    """Convert possibly compressed wire format into a Name.

    *parser* is a dns.wire.Parser.

    Raises ``dns.name.BadPointer`` if a compression pointer did not
    point backwards in the message.

    Raises ``dns.name.BadLabelType`` if an invalid label type was encountered.

    Returns a ``dns.name.Name``
    """

    labels = []
    biggest_pointer = parser.current
    with parser.restore_furthest():
        count = parser.get_uint8()
        while count != 0:
            if count < 64:
                labels.append(parser.get_bytes(count))
            elif count >= 192:
                current = (count & 0x3F) * 256 + parser.get_uint8()
                if current >= biggest_pointer:
                    raise BadPointer
                biggest_pointer = current
                parser.seek(current)
            else:
                raise BadLabelType
            count = parser.get_uint8()
        labels.append(b"")
    return Name(labels)


def from_wire(message: bytes, current: int) -> Tuple[Name, int]:
    """Convert possibly compressed wire format into a Name.

    *message* is a ``bytes`` containing an entire DNS message in DNS
    wire form.

    *current*, an ``int``, is the offset of the beginning of the name
    from the start of the message

    Raises ``dns.name.BadPointer`` if a compression pointer did not
    point backwards in the message.

    Raises ``dns.name.BadLabelType`` if an invalid label type was encountered.

    Returns a ``(dns.name.Name, int)`` tuple consisting of the name
    that was read and the number of bytes of the wire format message
    which were consumed reading it.
    """

    if not isinstance(message, bytes):
        raise ValueError("input to from_wire() must be a byte string")
    parser = dns.wire.Parser(message, current)
    name = from_wire_parser(parser)
    return (name, parser.current - current)
