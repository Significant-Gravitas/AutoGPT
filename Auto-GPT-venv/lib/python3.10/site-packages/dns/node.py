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

"""DNS nodes.  A node is a set of rdatasets."""

from typing import Any, Dict, Optional

import enum
import io

import dns.immutable
import dns.name
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.renderer


_cname_types = {
    dns.rdatatype.CNAME,
}

# "neutral" types can coexist with a CNAME and thus are not "other data"
_neutral_types = {
    dns.rdatatype.NSEC,  # RFC 4035 section 2.5
    dns.rdatatype.NSEC3,  # This is not likely to happen, but not impossible!
    dns.rdatatype.KEY,  # RFC 4035 section 2.5, RFC 3007
}


def _matches_type_or_its_signature(rdtypes, rdtype, covers):
    return rdtype in rdtypes or (rdtype == dns.rdatatype.RRSIG and covers in rdtypes)


@enum.unique
class NodeKind(enum.Enum):
    """Rdatasets in nodes"""

    REGULAR = 0  # a.k.a "other data"
    NEUTRAL = 1
    CNAME = 2

    @classmethod
    def classify(
        cls, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType
    ) -> "NodeKind":
        if _matches_type_or_its_signature(_cname_types, rdtype, covers):
            return NodeKind.CNAME
        elif _matches_type_or_its_signature(_neutral_types, rdtype, covers):
            return NodeKind.NEUTRAL
        else:
            return NodeKind.REGULAR

    @classmethod
    def classify_rdataset(cls, rdataset: dns.rdataset.Rdataset) -> "NodeKind":
        return cls.classify(rdataset.rdtype, rdataset.covers)


class Node:

    """A Node is a set of rdatasets.

    A node is either a CNAME node or an "other data" node.  A CNAME
    node contains only CNAME, KEY, NSEC, and NSEC3 rdatasets along with their
    covering RRSIG rdatasets.  An "other data" node contains any
    rdataset other than a CNAME or RRSIG(CNAME) rdataset.  When
    changes are made to a node, the CNAME or "other data" state is
    always consistent with the update, i.e. the most recent change
    wins.  For example, if you have a node which contains a CNAME
    rdataset, and then add an MX rdataset to it, then the CNAME
    rdataset will be deleted.  Likewise if you have a node containing
    an MX rdataset and add a CNAME rdataset, the MX rdataset will be
    deleted.
    """

    __slots__ = ["rdatasets"]

    def __init__(self):
        # the set of rdatasets, represented as a list.
        self.rdatasets = []

    def to_text(self, name: dns.name.Name, **kw: Dict[str, Any]) -> str:
        """Convert a node to text format.

        Each rdataset at the node is printed.  Any keyword arguments
        to this method are passed on to the rdataset's to_text() method.

        *name*, a ``dns.name.Name``, the owner name of the
        rdatasets.

        Returns a ``str``.

        """

        s = io.StringIO()
        for rds in self.rdatasets:
            if len(rds) > 0:
                s.write(rds.to_text(name, **kw))  # type: ignore[arg-type]
                s.write("\n")
        return s.getvalue()[:-1]

    def __repr__(self):
        return "<DNS node " + str(id(self)) + ">"

    def __eq__(self, other):
        #
        # This is inefficient.  Good thing we don't need to do it much.
        #
        for rd in self.rdatasets:
            if rd not in other.rdatasets:
                return False
        for rd in other.rdatasets:
            if rd not in self.rdatasets:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.rdatasets)

    def __iter__(self):
        return iter(self.rdatasets)

    def _append_rdataset(self, rdataset):
        """Append rdataset to the node with special handling for CNAME and
        other data conditions.

        Specifically, if the rdataset being appended has ``NodeKind.CNAME``,
        then all rdatasets other than KEY, NSEC, NSEC3, and their covering
        RRSIGs are deleted.  If the rdataset being appended has
        ``NodeKind.REGULAR`` then CNAME and RRSIG(CNAME) are deleted.
        """
        # Make having just one rdataset at the node fast.
        if len(self.rdatasets) > 0:
            kind = NodeKind.classify_rdataset(rdataset)
            if kind == NodeKind.CNAME:
                self.rdatasets = [
                    rds
                    for rds in self.rdatasets
                    if NodeKind.classify_rdataset(rds) != NodeKind.REGULAR
                ]
            elif kind == NodeKind.REGULAR:
                self.rdatasets = [
                    rds
                    for rds in self.rdatasets
                    if NodeKind.classify_rdataset(rds) != NodeKind.CNAME
                ]
            # Otherwise the rdataset is NodeKind.NEUTRAL and we do not need to
            # edit self.rdatasets.
        self.rdatasets.append(rdataset)

    def find_rdataset(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        create: bool = False,
    ) -> dns.rdataset.Rdataset:
        """Find an rdataset matching the specified properties in the
        current node.

        *rdclass*, a ``dns.rdataclass.RdataClass``, the class of the rdataset.

        *rdtype*, a ``dns.rdatatype.RdataType``, the type of the rdataset.

        *covers*, a ``dns.rdatatype.RdataType``, the covered type.
        Usually this value is ``dns.rdatatype.NONE``, but if the
        rdtype is ``dns.rdatatype.SIG`` or ``dns.rdatatype.RRSIG``,
        then the covers value will be the rdata type the SIG/RRSIG
        covers.  The library treats the SIG and RRSIG types as if they
        were a family of types, e.g. RRSIG(A), RRSIG(NS), RRSIG(SOA).
        This makes RRSIGs much easier to work with than if RRSIGs
        covering different rdata types were aggregated into a single
        RRSIG rdataset.

        *create*, a ``bool``.  If True, create the rdataset if it is not found.

        Raises ``KeyError`` if an rdataset of the desired type and class does
        not exist and *create* is not ``True``.

        Returns a ``dns.rdataset.Rdataset``.
        """

        for rds in self.rdatasets:
            if rds.match(rdclass, rdtype, covers):
                return rds
        if not create:
            raise KeyError
        rds = dns.rdataset.Rdataset(rdclass, rdtype, covers)
        self._append_rdataset(rds)
        return rds

    def get_rdataset(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        create: bool = False,
    ) -> Optional[dns.rdataset.Rdataset]:
        """Get an rdataset matching the specified properties in the
        current node.

        None is returned if an rdataset of the specified type and
        class does not exist and *create* is not ``True``.

        *rdclass*, an ``int``, the class of the rdataset.

        *rdtype*, an ``int``, the type of the rdataset.

        *covers*, an ``int``, the covered type.  Usually this value is
        dns.rdatatype.NONE, but if the rdtype is dns.rdatatype.SIG or
        dns.rdatatype.RRSIG, then the covers value will be the rdata
        type the SIG/RRSIG covers.  The library treats the SIG and RRSIG
        types as if they were a family of
        types, e.g. RRSIG(A), RRSIG(NS), RRSIG(SOA).  This makes RRSIGs much
        easier to work with than if RRSIGs covering different rdata
        types were aggregated into a single RRSIG rdataset.

        *create*, a ``bool``.  If True, create the rdataset if it is not found.

        Returns a ``dns.rdataset.Rdataset`` or ``None``.
        """

        try:
            rds = self.find_rdataset(rdclass, rdtype, covers, create)
        except KeyError:
            rds = None
        return rds

    def delete_rdataset(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
    ) -> None:
        """Delete the rdataset matching the specified properties in the
        current node.

        If a matching rdataset does not exist, it is not an error.

        *rdclass*, an ``int``, the class of the rdataset.

        *rdtype*, an ``int``, the type of the rdataset.

        *covers*, an ``int``, the covered type.
        """

        rds = self.get_rdataset(rdclass, rdtype, covers)
        if rds is not None:
            self.rdatasets.remove(rds)

    def replace_rdataset(self, replacement: dns.rdataset.Rdataset) -> None:
        """Replace an rdataset.

        It is not an error if there is no rdataset matching *replacement*.

        Ownership of the *replacement* object is transferred to the node;
        in other words, this method does not store a copy of *replacement*
        at the node, it stores *replacement* itself.

        *replacement*, a ``dns.rdataset.Rdataset``.

        Raises ``ValueError`` if *replacement* is not a
        ``dns.rdataset.Rdataset``.
        """

        if not isinstance(replacement, dns.rdataset.Rdataset):
            raise ValueError("replacement is not an rdataset")
        if isinstance(replacement, dns.rrset.RRset):
            # RRsets are not good replacements as the match() method
            # is not compatible.
            replacement = replacement.to_rdataset()
        self.delete_rdataset(
            replacement.rdclass, replacement.rdtype, replacement.covers
        )
        self._append_rdataset(replacement)

    def classify(self) -> NodeKind:
        """Classify a node.

        A node which contains a CNAME or RRSIG(CNAME) is a
        ``NodeKind.CNAME`` node.

        A node which contains only "neutral" types, i.e. types allowed to
        co-exist with a CNAME, is a ``NodeKind.NEUTRAL`` node.  The neutral
        types are NSEC, NSEC3, KEY, and their associated RRSIGS.  An empty node
        is also considered neutral.

        A node which contains some rdataset which is not a CNAME, RRSIG(CNAME),
        or a neutral type is a a ``NodeKind.REGULAR`` node.  Regular nodes are
        also commonly referred to as "other data".
        """
        for rdataset in self.rdatasets:
            kind = NodeKind.classify(rdataset.rdtype, rdataset.covers)
            if kind != NodeKind.NEUTRAL:
                return kind
        return NodeKind.NEUTRAL

    def is_immutable(self) -> bool:
        return False


@dns.immutable.immutable
class ImmutableNode(Node):
    def __init__(self, node):
        super().__init__()
        self.rdatasets = tuple(
            [dns.rdataset.ImmutableRdataset(rds) for rds in node.rdatasets]
        )

    def find_rdataset(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        create: bool = False,
    ) -> dns.rdataset.Rdataset:
        if create:
            raise TypeError("immutable")
        return super().find_rdataset(rdclass, rdtype, covers, False)

    def get_rdataset(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        create: bool = False,
    ) -> Optional[dns.rdataset.Rdataset]:
        if create:
            raise TypeError("immutable")
        return super().get_rdataset(rdclass, rdtype, covers, False)

    def delete_rdataset(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
    ) -> None:
        raise TypeError("immutable")

    def replace_rdataset(self, replacement: dns.rdataset.Rdataset) -> None:
        raise TypeError("immutable")

    def is_immutable(self) -> bool:
        return True
