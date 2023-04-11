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

"""DNS rdatasets (an rdataset is a set of rdatas of a given type and class)"""

from typing import Any, cast, Collection, Dict, List, Optional, Union

import io
import random
import struct

import dns.exception
import dns.immutable
import dns.name
import dns.rdatatype
import dns.rdataclass
import dns.rdata
import dns.set
import dns.ttl

# define SimpleSet here for backwards compatibility
SimpleSet = dns.set.Set


class DifferingCovers(dns.exception.DNSException):
    """An attempt was made to add a DNS SIG/RRSIG whose covered type
    is not the same as that of the other rdatas in the rdataset."""


class IncompatibleTypes(dns.exception.DNSException):
    """An attempt was made to add DNS RR data of an incompatible type."""


class Rdataset(dns.set.Set):

    """A DNS rdataset."""

    __slots__ = ["rdclass", "rdtype", "covers", "ttl"]

    def __init__(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        ttl: int = 0,
    ):
        """Create a new rdataset of the specified class and type.

        *rdclass*, a ``dns.rdataclass.RdataClass``, the rdataclass.

        *rdtype*, an ``dns.rdatatype.RdataType``, the rdatatype.

        *covers*, an ``dns.rdatatype.RdataType``, the covered rdatatype.

        *ttl*, an ``int``, the TTL.
        """

        super().__init__()
        self.rdclass = rdclass
        self.rdtype: dns.rdatatype.RdataType = rdtype
        self.covers: dns.rdatatype.RdataType = covers
        self.ttl = ttl

    def _clone(self):
        obj = super()._clone()
        obj.rdclass = self.rdclass
        obj.rdtype = self.rdtype
        obj.covers = self.covers
        obj.ttl = self.ttl
        return obj

    def update_ttl(self, ttl: int) -> None:
        """Perform TTL minimization.

        Set the TTL of the rdataset to be the lesser of the set's current
        TTL or the specified TTL.  If the set contains no rdatas, set the TTL
        to the specified TTL.

        *ttl*, an ``int`` or ``str``.
        """
        ttl = dns.ttl.make(ttl)
        if len(self) == 0:
            self.ttl = ttl
        elif ttl < self.ttl:
            self.ttl = ttl

    def add(  # pylint: disable=arguments-differ,arguments-renamed
        self, rd: dns.rdata.Rdata, ttl: Optional[int] = None
    ) -> None:
        """Add the specified rdata to the rdataset.

        If the optional *ttl* parameter is supplied, then
        ``self.update_ttl(ttl)`` will be called prior to adding the rdata.

        *rd*, a ``dns.rdata.Rdata``, the rdata

        *ttl*, an ``int``, the TTL.

        Raises ``dns.rdataset.IncompatibleTypes`` if the type and class
        do not match the type and class of the rdataset.

        Raises ``dns.rdataset.DifferingCovers`` if the type is a signature
        type and the covered type does not match that of the rdataset.
        """

        #
        # If we're adding a signature, do some special handling to
        # check that the signature covers the same type as the
        # other rdatas in this rdataset.  If this is the first rdata
        # in the set, initialize the covers field.
        #
        if self.rdclass != rd.rdclass or self.rdtype != rd.rdtype:
            raise IncompatibleTypes
        if ttl is not None:
            self.update_ttl(ttl)
        if self.rdtype == dns.rdatatype.RRSIG or self.rdtype == dns.rdatatype.SIG:
            covers = rd.covers()
            if len(self) == 0 and self.covers == dns.rdatatype.NONE:
                self.covers = covers
            elif self.covers != covers:
                raise DifferingCovers
        if dns.rdatatype.is_singleton(rd.rdtype) and len(self) > 0:
            self.clear()
        super().add(rd)

    def union_update(self, other):
        self.update_ttl(other.ttl)
        super().union_update(other)

    def intersection_update(self, other):
        self.update_ttl(other.ttl)
        super().intersection_update(other)

    def update(self, other):
        """Add all rdatas in other to self.

        *other*, a ``dns.rdataset.Rdataset``, the rdataset from which
        to update.
        """

        self.update_ttl(other.ttl)
        super().update(other)

    def _rdata_repr(self):
        def maybe_truncate(s):
            if len(s) > 100:
                return s[:100] + "..."
            return s

        return "[%s]" % ", ".join("<%s>" % maybe_truncate(str(rr)) for rr in self)

    def __repr__(self):
        if self.covers == 0:
            ctext = ""
        else:
            ctext = "(" + dns.rdatatype.to_text(self.covers) + ")"
        return (
            "<DNS "
            + dns.rdataclass.to_text(self.rdclass)
            + " "
            + dns.rdatatype.to_text(self.rdtype)
            + ctext
            + " rdataset: "
            + self._rdata_repr()
            + ">"
        )

    def __str__(self):
        return self.to_text()

    def __eq__(self, other):
        if not isinstance(other, Rdataset):
            return False
        if (
            self.rdclass != other.rdclass
            or self.rdtype != other.rdtype
            or self.covers != other.covers
        ):
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_text(
        self,
        name: Optional[dns.name.Name] = None,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        override_rdclass: Optional[dns.rdataclass.RdataClass] = None,
        want_comments: bool = False,
        **kw: Dict[str, Any],
    ) -> str:
        """Convert the rdataset into DNS zone file format.

        See ``dns.name.Name.choose_relativity`` for more information
        on how *origin* and *relativize* determine the way names
        are emitted.

        Any additional keyword arguments are passed on to the rdata
        ``to_text()`` method.

        *name*, a ``dns.name.Name``.  If name is not ``None``, emit RRs with
        *name* as the owner name.

        *origin*, a ``dns.name.Name`` or ``None``, the origin for relative
        names.

        *relativize*, a ``bool``.  If ``True``, names will be relativized
        to *origin*.

        *override_rdclass*, a ``dns.rdataclass.RdataClass`` or ``None``.
        If not ``None``, use this class instead of the Rdataset's class.

        *want_comments*, a ``bool``.  If ``True``, emit comments for rdata
        which have them.  The default is ``False``.
        """

        if name is not None:
            name = name.choose_relativity(origin, relativize)
            ntext = str(name)
            pad = " "
        else:
            ntext = ""
            pad = ""
        s = io.StringIO()
        if override_rdclass is not None:
            rdclass = override_rdclass
        else:
            rdclass = self.rdclass
        if len(self) == 0:
            #
            # Empty rdatasets are used for the question section, and in
            # some dynamic updates, so we don't need to print out the TTL
            # (which is meaningless anyway).
            #
            s.write(
                "{}{}{} {}\n".format(
                    ntext,
                    pad,
                    dns.rdataclass.to_text(rdclass),
                    dns.rdatatype.to_text(self.rdtype),
                )
            )
        else:
            for rd in self:
                extra = ""
                if want_comments:
                    if rd.rdcomment:
                        extra = f" ;{rd.rdcomment}"
                s.write(
                    "%s%s%d %s %s %s%s\n"
                    % (
                        ntext,
                        pad,
                        self.ttl,
                        dns.rdataclass.to_text(rdclass),
                        dns.rdatatype.to_text(self.rdtype),
                        rd.to_text(origin=origin, relativize=relativize, **kw),
                        extra,
                    )
                )
        #
        # We strip off the final \n for the caller's convenience in printing
        #
        return s.getvalue()[:-1]

    def to_wire(
        self,
        name: dns.name.Name,
        file: Any,
        compress: Optional[dns.name.CompressType] = None,
        origin: Optional[dns.name.Name] = None,
        override_rdclass: Optional[dns.rdataclass.RdataClass] = None,
        want_shuffle: bool = True,
    ) -> int:
        """Convert the rdataset to wire format.

        *name*, a ``dns.name.Name`` is the owner name to use.

        *file* is the file where the name is emitted (typically a
        BytesIO file).

        *compress*, a ``dict``, is the compression table to use.  If
        ``None`` (the default), names will not be compressed.

        *origin* is a ``dns.name.Name`` or ``None``.  If the name is
        relative and origin is not ``None``, then *origin* will be appended
        to it.

        *override_rdclass*, an ``int``, is used as the class instead of the
        class of the rdataset.  This is useful when rendering rdatasets
        associated with dynamic updates.

        *want_shuffle*, a ``bool``.  If ``True``, then the order of the
        Rdatas within the Rdataset will be shuffled before rendering.

        Returns an ``int``, the number of records emitted.
        """

        if override_rdclass is not None:
            rdclass = override_rdclass
            want_shuffle = False
        else:
            rdclass = self.rdclass
        file.seek(0, io.SEEK_END)
        if len(self) == 0:
            name.to_wire(file, compress, origin)
            stuff = struct.pack("!HHIH", self.rdtype, rdclass, 0, 0)
            file.write(stuff)
            return 1
        else:
            l: Union[Rdataset, List[dns.rdata.Rdata]]
            if want_shuffle:
                l = list(self)
                random.shuffle(l)
            else:
                l = self
            for rd in l:
                name.to_wire(file, compress, origin)
                stuff = struct.pack("!HHIH", self.rdtype, rdclass, self.ttl, 0)
                file.write(stuff)
                start = file.tell()
                rd.to_wire(file, compress, origin)
                end = file.tell()
                assert end - start < 65536
                file.seek(start - 2)
                stuff = struct.pack("!H", end - start)
                file.write(stuff)
                file.seek(0, io.SEEK_END)
            return len(self)

    def match(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType,
    ) -> bool:
        """Returns ``True`` if this rdataset matches the specified class,
        type, and covers.
        """
        if self.rdclass == rdclass and self.rdtype == rdtype and self.covers == covers:
            return True
        return False

    def processing_order(self) -> List[dns.rdata.Rdata]:
        """Return rdatas in a valid processing order according to the type's
        specification.  For example, MX records are in preference order from
        lowest to highest preferences, with items of the same preference
        shuffled.

        For types that do not define a processing order, the rdatas are
        simply shuffled.
        """
        if len(self) == 0:
            return []
        else:
            return self[0]._processing_order(iter(self))


@dns.immutable.immutable
class ImmutableRdataset(Rdataset):  # lgtm[py/missing-equals]

    """An immutable DNS rdataset."""

    _clone_class = Rdataset

    def __init__(self, rdataset: Rdataset):
        """Create an immutable rdataset from the specified rdataset."""

        super().__init__(
            rdataset.rdclass, rdataset.rdtype, rdataset.covers, rdataset.ttl
        )
        self.items = dns.immutable.Dict(rdataset.items)

    def update_ttl(self, ttl):
        raise TypeError("immutable")

    def add(self, rd, ttl=None):
        raise TypeError("immutable")

    def union_update(self, other):
        raise TypeError("immutable")

    def intersection_update(self, other):
        raise TypeError("immutable")

    def update(self, other):
        raise TypeError("immutable")

    def __delitem__(self, i):
        raise TypeError("immutable")

    # lgtm complains about these not raising ArithmeticError, but there is
    # precedent for overrides of these methods in other classes to raise
    # TypeError, and it seems like the better exception.

    def __ior__(self, other):  # lgtm[py/unexpected-raise-in-special-method]
        raise TypeError("immutable")

    def __iand__(self, other):  # lgtm[py/unexpected-raise-in-special-method]
        raise TypeError("immutable")

    def __iadd__(self, other):  # lgtm[py/unexpected-raise-in-special-method]
        raise TypeError("immutable")

    def __isub__(self, other):  # lgtm[py/unexpected-raise-in-special-method]
        raise TypeError("immutable")

    def clear(self):
        raise TypeError("immutable")

    def __copy__(self):
        return ImmutableRdataset(super().copy())

    def copy(self):
        return ImmutableRdataset(super().copy())

    def union(self, other):
        return ImmutableRdataset(super().union(other))

    def intersection(self, other):
        return ImmutableRdataset(super().intersection(other))

    def difference(self, other):
        return ImmutableRdataset(super().difference(other))

    def symmetric_difference(self, other):
        return ImmutableRdataset(super().symmetric_difference(other))


def from_text_list(
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    ttl: int,
    text_rdatas: Collection[str],
    idna_codec: Optional[dns.name.IDNACodec] = None,
    origin: Optional[dns.name.Name] = None,
    relativize: bool = True,
    relativize_to: Optional[dns.name.Name] = None,
) -> Rdataset:
    """Create an rdataset with the specified class, type, and TTL, and with
    the specified list of rdatas in text format.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder to use; if ``None``, the default IDNA 2003
    encoder/decoder is used.

    *origin*, a ``dns.name.Name`` (or ``None``), the
    origin to use for relative names.

    *relativize*, a ``bool``.  If true, name will be relativized.

    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use
    when relativizing names.  If not set, the *origin* value will be used.

    Returns a ``dns.rdataset.Rdataset`` object.
    """

    the_rdclass = dns.rdataclass.RdataClass.make(rdclass)
    the_rdtype = dns.rdatatype.RdataType.make(rdtype)
    r = Rdataset(the_rdclass, the_rdtype)
    r.update_ttl(ttl)
    for t in text_rdatas:
        rd = dns.rdata.from_text(
            r.rdclass, r.rdtype, t, origin, relativize, relativize_to, idna_codec
        )
        r.add(rd)
    return r


def from_text(
    rdclass: Union[dns.rdataclass.RdataClass, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    ttl: int,
    *text_rdatas: Any,
) -> Rdataset:
    """Create an rdataset with the specified class, type, and TTL, and with
    the specified rdatas in text format.

    Returns a ``dns.rdataset.Rdataset`` object.
    """

    return from_text_list(rdclass, rdtype, ttl, cast(Collection[str], text_rdatas))


def from_rdata_list(ttl: int, rdatas: Collection[dns.rdata.Rdata]) -> Rdataset:
    """Create an rdataset with the specified TTL, and with
    the specified list of rdata objects.

    Returns a ``dns.rdataset.Rdataset`` object.
    """

    if len(rdatas) == 0:
        raise ValueError("rdata list must not be empty")
    r = None
    for rd in rdatas:
        if r is None:
            r = Rdataset(rd.rdclass, rd.rdtype)
            r.update_ttl(ttl)
        r.add(rd)
    assert r is not None
    return r


def from_rdata(ttl: int, *rdatas: Any) -> Rdataset:
    """Create an rdataset with the specified TTL, and with
    the specified rdata objects.

    Returns a ``dns.rdataset.Rdataset`` object.
    """

    return from_rdata_list(ttl, cast(Collection[dns.rdata.Rdata], rdatas))
