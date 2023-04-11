# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2017 Nominum, Inc.
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

"""DNS RRsets (an RRset is a named rdataset)"""

from typing import Any, cast, Collection, Dict, Optional, Union

import dns.name
import dns.rdataset
import dns.rdataclass
import dns.renderer


class RRset(dns.rdataset.Rdataset):

    """A DNS RRset (named rdataset).

    RRset inherits from Rdataset, and RRsets can be treated as
    Rdatasets in most cases.  There are, however, a few notable
    exceptions.  RRsets have different to_wire() and to_text() method
    arguments, reflecting the fact that RRsets always have an owner
    name.
    """

    __slots__ = ["name", "deleting"]

    def __init__(
        self,
        name: dns.name.Name,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        deleting: Optional[dns.rdataclass.RdataClass] = None,
    ):
        """Create a new RRset."""

        super().__init__(rdclass, rdtype, covers)
        self.name = name
        self.deleting = deleting

    def _clone(self):
        obj = super()._clone()
        obj.name = self.name
        obj.deleting = self.deleting
        return obj

    def __repr__(self):
        if self.covers == 0:
            ctext = ""
        else:
            ctext = "(" + dns.rdatatype.to_text(self.covers) + ")"
        if self.deleting is not None:
            dtext = " delete=" + dns.rdataclass.to_text(self.deleting)
        else:
            dtext = ""
        return (
            "<DNS "
            + str(self.name)
            + " "
            + dns.rdataclass.to_text(self.rdclass)
            + " "
            + dns.rdatatype.to_text(self.rdtype)
            + ctext
            + dtext
            + " RRset: "
            + self._rdata_repr()
            + ">"
        )

    def __str__(self):
        return self.to_text()

    def __eq__(self, other):
        if isinstance(other, RRset):
            if self.name != other.name:
                return False
        elif not isinstance(other, dns.rdataset.Rdataset):
            return False
        return super().__eq__(other)

    def match(self, *args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        """Does this rrset match the specified attributes?

        Behaves as :py:func:`full_match()` if the first argument is a
        ``dns.name.Name``, and as :py:func:`dns.rdataset.Rdataset.match()`
        otherwise.

        (This behavior fixes a design mistake where the signature of this
        method became incompatible with that of its superclass.  The fix
        makes RRsets matchable as Rdatasets while preserving backwards
        compatibility.)
        """
        if isinstance(args[0], dns.name.Name):
            return self.full_match(*args, **kwargs)  # type: ignore[arg-type]
        else:
            return super().match(*args, **kwargs)  # type: ignore[arg-type]

    def full_match(
        self,
        name: dns.name.Name,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType,
        deleting: Optional[dns.rdataclass.RdataClass] = None,
    ) -> bool:
        """Returns ``True`` if this rrset matches the specified name, class,
        type, covers, and deletion state.
        """
        if not super().match(rdclass, rdtype, covers):
            return False
        if self.name != name or self.deleting != deleting:
            return False
        return True

    # pylint: disable=arguments-differ

    def to_text(  # type: ignore[override]
        self,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        **kw: Dict[str, Any]
    ) -> str:
        """Convert the RRset into DNS zone file format.

        See ``dns.name.Name.choose_relativity`` for more information
        on how *origin* and *relativize* determine the way names
        are emitted.

        Any additional keyword arguments are passed on to the rdata
        ``to_text()`` method.

        *origin*, a ``dns.name.Name`` or ``None``, the origin for relative
        names.

        *relativize*, a ``bool``.  If ``True``, names will be relativized
        to *origin*.
        """

        return super().to_text(
            self.name, origin, relativize, self.deleting, **kw  # type: ignore
        )

    def to_wire(  # type: ignore[override]
        self,
        file: Any,
        compress: Optional[dns.name.CompressType] = None,  # type: ignore
        origin: Optional[dns.name.Name] = None,
        **kw: Dict[str, Any]
    ) -> int:
        """Convert the RRset to wire format.

        All keyword arguments are passed to ``dns.rdataset.to_wire()``; see
        that function for details.

        Returns an ``int``, the number of records emitted.
        """

        return super().to_wire(
            self.name, file, compress, origin, self.deleting, **kw  # type:ignore
        )

    # pylint: enable=arguments-differ

    def to_rdataset(self) -> dns.rdataset.Rdataset:
        """Convert an RRset into an Rdataset.

        Returns a ``dns.rdataset.Rdataset``.
        """
        return dns.rdataset.from_rdata_list(self.ttl, list(self))


def from_text_list(
    name: Union[dns.name.Name, str],
    ttl: int,
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    text_rdatas: Collection[str],
    idna_codec: Optional[dns.name.IDNACodec] = None,
    origin: Optional[dns.name.Name] = None,
    relativize: bool = True,
    relativize_to: Optional[dns.name.Name] = None,
) -> RRset:
    """Create an RRset with the specified name, TTL, class, and type, and with
    the specified list of rdatas in text format.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder to use; if ``None``, the default IDNA 2003
    encoder/decoder is used.

    *origin*, a ``dns.name.Name`` (or ``None``), the
    origin to use for relative names.

    *relativize*, a ``bool``.  If true, name will be relativized.

    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use
    when relativizing names.  If not set, the *origin* value will be used.

    Returns a ``dns.rrset.RRset`` object.
    """

    if isinstance(name, str):
        name = dns.name.from_text(name, None, idna_codec=idna_codec)
    the_rdclass = dns.rdataclass.RdataClass.make(rdclass)
    the_rdtype = dns.rdatatype.RdataType.make(rdtype)
    r = RRset(name, the_rdclass, the_rdtype)
    r.update_ttl(ttl)
    for t in text_rdatas:
        rd = dns.rdata.from_text(
            r.rdclass, r.rdtype, t, origin, relativize, relativize_to, idna_codec
        )
        r.add(rd)
    return r


def from_text(
    name: Union[dns.name.Name, str],
    ttl: int,
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    *text_rdatas: Any
) -> RRset:
    """Create an RRset with the specified name, TTL, class, and type and with
    the specified rdatas in text format.

    Returns a ``dns.rrset.RRset`` object.
    """

    return from_text_list(
        name, ttl, rdclass, rdtype, cast(Collection[str], text_rdatas)
    )


def from_rdata_list(
    name: Union[dns.name.Name, str],
    ttl: int,
    rdatas: Collection[dns.rdata.Rdata],
    idna_codec: Optional[dns.name.IDNACodec] = None,
) -> RRset:
    """Create an RRset with the specified name and TTL, and with
    the specified list of rdata objects.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder to use; if ``None``, the default IDNA 2003
    encoder/decoder is used.

    Returns a ``dns.rrset.RRset`` object.

    """

    if isinstance(name, str):
        name = dns.name.from_text(name, None, idna_codec=idna_codec)

    if len(rdatas) == 0:
        raise ValueError("rdata list must not be empty")
    r = None
    for rd in rdatas:
        if r is None:
            r = RRset(name, rd.rdclass, rd.rdtype)
            r.update_ttl(ttl)
        r.add(rd)
    assert r is not None
    return r


def from_rdata(name: Union[dns.name.Name, str], ttl: int, *rdatas: Any) -> RRset:
    """Create an RRset with the specified name and TTL, and with
    the specified rdata objects.

    Returns a ``dns.rrset.RRset`` object.
    """

    return from_rdata_list(name, ttl, cast(Collection[dns.rdata.Rdata], rdatas))
