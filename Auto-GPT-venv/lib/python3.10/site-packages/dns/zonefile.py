# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2007, 2009-2011 Nominum, Inc.
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

"""DNS Zones."""

from typing import Any, Iterable, List, Optional, Set, Tuple, Union

import re
import sys

import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdatatype
import dns.rdata
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
import dns.grange


class UnknownOrigin(dns.exception.DNSException):
    """Unknown origin"""


class CNAMEAndOtherData(dns.exception.DNSException):
    """A node has a CNAME and other data"""


def _check_cname_and_other_data(txn, name, rdataset):
    rdataset_kind = dns.node.NodeKind.classify_rdataset(rdataset)
    node = txn.get_node(name)
    if node is None:
        # empty nodes are neutral.
        return
    node_kind = node.classify()
    if (
        node_kind == dns.node.NodeKind.CNAME
        and rdataset_kind == dns.node.NodeKind.REGULAR
    ):
        raise CNAMEAndOtherData("rdataset type is not compatible with a CNAME node")
    elif (
        node_kind == dns.node.NodeKind.REGULAR
        and rdataset_kind == dns.node.NodeKind.CNAME
    ):
        raise CNAMEAndOtherData(
            "CNAME rdataset is not compatible with a regular data node"
        )
    # Otherwise at least one of the node and the rdataset is neutral, so
    # adding the rdataset is ok


SavedStateType = Tuple[
    dns.tokenizer.Tokenizer,
    Optional[dns.name.Name],  # current_origin
    Optional[dns.name.Name],  # last_name
    Optional[Any],  # current_file
    int,  # last_ttl
    bool,  # last_ttl_known
    int,  # default_ttl
    bool,
]  # default_ttl_known


def _upper_dollarize(s):
    s = s.upper()
    if not s.startswith("$"):
        s = "$" + s
    return s


class Reader:

    """Read a DNS zone file into a transaction."""

    def __init__(
        self,
        tok: dns.tokenizer.Tokenizer,
        rdclass: dns.rdataclass.RdataClass,
        txn: dns.transaction.Transaction,
        allow_include: bool = False,
        allow_directives: Union[bool, Iterable[str]] = True,
        force_name: Optional[dns.name.Name] = None,
        force_ttl: Optional[int] = None,
        force_rdclass: Optional[dns.rdataclass.RdataClass] = None,
        force_rdtype: Optional[dns.rdatatype.RdataType] = None,
        default_ttl: Optional[int] = None,
    ):
        self.tok = tok
        (self.zone_origin, self.relativize, _) = txn.manager.origin_information()
        self.current_origin = self.zone_origin
        self.last_ttl = 0
        self.last_ttl_known = False
        if force_ttl is not None:
            default_ttl = force_ttl
        if default_ttl is None:
            self.default_ttl = 0
            self.default_ttl_known = False
        else:
            self.default_ttl = default_ttl
            self.default_ttl_known = True
        self.last_name = self.current_origin
        self.zone_rdclass = rdclass
        self.txn = txn
        self.saved_state: List[SavedStateType] = []
        self.current_file: Optional[Any] = None
        self.allowed_directives: Set[str]
        if allow_directives is True:
            self.allowed_directives = {"$GENERATE", "$ORIGIN", "$TTL"}
            if allow_include:
                self.allowed_directives.add("$INCLUDE")
        elif allow_directives is False:
            # allow_include was ignored in earlier releases if allow_directives was
            # False, so we continue that.
            self.allowed_directives = set()
        else:
            # Note that if directives are explicitly specified, then allow_include
            # is ignored.
            self.allowed_directives = set(_upper_dollarize(d) for d in allow_directives)
        self.force_name = force_name
        self.force_ttl = force_ttl
        self.force_rdclass = force_rdclass
        self.force_rdtype = force_rdtype
        self.txn.check_put_rdataset(_check_cname_and_other_data)

    def _eat_line(self):
        while 1:
            token = self.tok.get()
            if token.is_eol_or_eof():
                break

    def _get_identifier(self):
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
        return token

    def _rr_line(self):
        """Process one line from a DNS zone file."""
        token = None
        # Name
        if self.force_name is not None:
            name = self.force_name
        else:
            if self.current_origin is None:
                raise UnknownOrigin
            token = self.tok.get(want_leading=True)
            if not token.is_whitespace():
                self.last_name = self.tok.as_name(token, self.current_origin)
            else:
                token = self.tok.get()
                if token.is_eol_or_eof():
                    # treat leading WS followed by EOL/EOF as if they were EOL/EOF.
                    return
                self.tok.unget(token)
            name = self.last_name
            if not name.is_subdomain(self.zone_origin):
                self._eat_line()
                return
            if self.relativize:
                name = name.relativize(self.zone_origin)

        # TTL
        if self.force_ttl is not None:
            ttl = self.force_ttl
            self.last_ttl = ttl
            self.last_ttl_known = True
        else:
            token = self._get_identifier()
            ttl = None
            try:
                ttl = dns.ttl.from_text(token.value)
                self.last_ttl = ttl
                self.last_ttl_known = True
                token = None
            except dns.ttl.BadTTL:
                if self.default_ttl_known:
                    ttl = self.default_ttl
                elif self.last_ttl_known:
                    ttl = self.last_ttl
                self.tok.unget(token)

        # Class
        if self.force_rdclass is not None:
            rdclass = self.force_rdclass
        else:
            token = self._get_identifier()
            try:
                rdclass = dns.rdataclass.from_text(token.value)
            except dns.exception.SyntaxError:
                raise
            except Exception:
                rdclass = self.zone_rdclass
                self.tok.unget(token)
            if rdclass != self.zone_rdclass:
                raise dns.exception.SyntaxError("RR class is not zone's class")

        # Type
        if self.force_rdtype is not None:
            rdtype = self.force_rdtype
        else:
            token = self._get_identifier()
            try:
                rdtype = dns.rdatatype.from_text(token.value)
            except Exception:
                raise dns.exception.SyntaxError("unknown rdatatype '%s'" % token.value)

        try:
            rd = dns.rdata.from_text(
                rdclass,
                rdtype,
                self.tok,
                self.current_origin,
                self.relativize,
                self.zone_origin,
            )
        except dns.exception.SyntaxError:
            # Catch and reraise.
            raise
        except Exception:
            # All exceptions that occur in the processing of rdata
            # are treated as syntax errors.  This is not strictly
            # correct, but it is correct almost all of the time.
            # We convert them to syntax errors so that we can emit
            # helpful filename:line info.
            (ty, va) = sys.exc_info()[:2]
            raise dns.exception.SyntaxError(
                "caught exception {}: {}".format(str(ty), str(va))
            )

        if not self.default_ttl_known and rdtype == dns.rdatatype.SOA:
            # The pre-RFC2308 and pre-BIND9 behavior inherits the zone default
            # TTL from the SOA minttl if no $TTL statement is present before the
            # SOA is parsed.
            self.default_ttl = rd.minimum
            self.default_ttl_known = True
            if ttl is None:
                # if we didn't have a TTL on the SOA, set it!
                ttl = rd.minimum

        # TTL check.  We had to wait until now to do this as the SOA RR's
        # own TTL can be inferred from its minimum.
        if ttl is None:
            raise dns.exception.SyntaxError("Missing default TTL value")

        self.txn.add(name, ttl, rd)

    def _parse_modify(self, side: str) -> Tuple[str, str, int, int, str]:
        # Here we catch everything in '{' '}' in a group so we can replace it
        # with ''.
        is_generate1 = re.compile(r"^.*\$({(\+|-?)(\d+),(\d+),(.)}).*$")
        is_generate2 = re.compile(r"^.*\$({(\+|-?)(\d+)}).*$")
        is_generate3 = re.compile(r"^.*\$({(\+|-?)(\d+),(\d+)}).*$")
        # Sometimes there are modifiers in the hostname. These come after
        # the dollar sign. They are in the form: ${offset[,width[,base]]}.
        # Make names
        g1 = is_generate1.match(side)
        if g1:
            mod, sign, offset, width, base = g1.groups()
            if sign == "":
                sign = "+"
        g2 = is_generate2.match(side)
        if g2:
            mod, sign, offset = g2.groups()
            if sign == "":
                sign = "+"
            width = 0
            base = "d"
        g3 = is_generate3.match(side)
        if g3:
            mod, sign, offset, width = g3.groups()
            if sign == "":
                sign = "+"
            base = "d"

        if not (g1 or g2 or g3):
            mod = ""
            sign = "+"
            offset = 0
            width = 0
            base = "d"

        offset = int(offset)
        width = int(width)

        if sign not in ["+", "-"]:
            raise dns.exception.SyntaxError("invalid offset sign %s" % sign)
        if base not in ["d", "o", "x", "X", "n", "N"]:
            raise dns.exception.SyntaxError("invalid type %s" % base)

        return mod, sign, offset, width, base

    def _generate_line(self):
        # range lhs [ttl] [class] type rhs [ comment ]
        """Process one line containing the GENERATE statement from a DNS
        zone file."""
        if self.current_origin is None:
            raise UnknownOrigin

        token = self.tok.get()
        # Range (required)
        try:
            start, stop, step = dns.grange.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except Exception:
            raise dns.exception.SyntaxError

        # lhs (required)
        try:
            lhs = token.value
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except Exception:
            raise dns.exception.SyntaxError

        # TTL
        try:
            ttl = dns.ttl.from_text(token.value)
            self.last_ttl = ttl
            self.last_ttl_known = True
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.ttl.BadTTL:
            if not (self.last_ttl_known or self.default_ttl_known):
                raise dns.exception.SyntaxError("Missing default TTL value")
            if self.default_ttl_known:
                ttl = self.default_ttl
            elif self.last_ttl_known:
                ttl = self.last_ttl
        # Class
        try:
            rdclass = dns.rdataclass.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.exception.SyntaxError:
            raise dns.exception.SyntaxError
        except Exception:
            rdclass = self.zone_rdclass
        if rdclass != self.zone_rdclass:
            raise dns.exception.SyntaxError("RR class is not zone's class")
        # Type
        try:
            rdtype = dns.rdatatype.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except Exception:
            raise dns.exception.SyntaxError("unknown rdatatype '%s'" % token.value)

        # rhs (required)
        rhs = token.value

        def _calculate_index(counter: int, offset_sign: str, offset: int) -> int:
            """Calculate the index from the counter and offset."""
            if offset_sign == "-":
                offset *= -1
            return counter + offset

        def _format_index(index: int, base: str, width: int) -> str:
            """Format the index with the given base, and zero-fill it
            to the given width."""
            if base in ["d", "o", "x", "X"]:
                return format(index, base).zfill(width)

            # base can only be n or N here
            hexa = _format_index(index, "x", width)
            nibbles = ".".join(hexa[::-1])[:width]
            if base == "N":
                nibbles = nibbles.upper()
            return nibbles

        lmod, lsign, loffset, lwidth, lbase = self._parse_modify(lhs)
        rmod, rsign, roffset, rwidth, rbase = self._parse_modify(rhs)
        for i in range(start, stop + 1, step):
            # +1 because bind is inclusive and python is exclusive

            lindex = _calculate_index(i, lsign, loffset)
            rindex = _calculate_index(i, rsign, roffset)

            lzfindex = _format_index(lindex, lbase, lwidth)
            rzfindex = _format_index(rindex, rbase, rwidth)

            name = lhs.replace("$%s" % (lmod), lzfindex)
            rdata = rhs.replace("$%s" % (rmod), rzfindex)

            self.last_name = dns.name.from_text(
                name, self.current_origin, self.tok.idna_codec
            )
            name = self.last_name
            if not name.is_subdomain(self.zone_origin):
                self._eat_line()
                return
            if self.relativize:
                name = name.relativize(self.zone_origin)

            try:
                rd = dns.rdata.from_text(
                    rdclass,
                    rdtype,
                    rdata,
                    self.current_origin,
                    self.relativize,
                    self.zone_origin,
                )
            except dns.exception.SyntaxError:
                # Catch and reraise.
                raise
            except Exception:
                # All exceptions that occur in the processing of rdata
                # are treated as syntax errors.  This is not strictly
                # correct, but it is correct almost all of the time.
                # We convert them to syntax errors so that we can emit
                # helpful filename:line info.
                (ty, va) = sys.exc_info()[:2]
                raise dns.exception.SyntaxError(
                    "caught exception %s: %s" % (str(ty), str(va))
                )

            self.txn.add(name, ttl, rd)

    def read(self) -> None:
        """Read a DNS zone file and build a zone object.

        @raises dns.zone.NoSOA: No SOA RR was found at the zone origin
        @raises dns.zone.NoNS: No NS RRset was found at the zone origin
        """

        try:
            while 1:
                token = self.tok.get(True, True)
                if token.is_eof():
                    if self.current_file is not None:
                        self.current_file.close()
                    if len(self.saved_state) > 0:
                        (
                            self.tok,
                            self.current_origin,
                            self.last_name,
                            self.current_file,
                            self.last_ttl,
                            self.last_ttl_known,
                            self.default_ttl,
                            self.default_ttl_known,
                        ) = self.saved_state.pop(-1)
                        continue
                    break
                elif token.is_eol():
                    continue
                elif token.is_comment():
                    self.tok.get_eol()
                    continue
                elif token.value[0] == "$" and len(self.allowed_directives) > 0:
                    # Note that we only run directive processing code if at least
                    # one directive is allowed in order to be backwards compatible
                    c = token.value.upper()
                    if c not in self.allowed_directives:
                        raise dns.exception.SyntaxError(
                            f"zone file directive '{c}' is not allowed"
                        )
                    if c == "$TTL":
                        token = self.tok.get()
                        if not token.is_identifier():
                            raise dns.exception.SyntaxError("bad $TTL")
                        self.default_ttl = dns.ttl.from_text(token.value)
                        self.default_ttl_known = True
                        self.tok.get_eol()
                    elif c == "$ORIGIN":
                        self.current_origin = self.tok.get_name()
                        self.tok.get_eol()
                        if self.zone_origin is None:
                            self.zone_origin = self.current_origin
                        self.txn._set_origin(self.current_origin)
                    elif c == "$INCLUDE":
                        token = self.tok.get()
                        filename = token.value
                        token = self.tok.get()
                        new_origin: Optional[dns.name.Name]
                        if token.is_identifier():
                            new_origin = dns.name.from_text(
                                token.value, self.current_origin, self.tok.idna_codec
                            )
                            self.tok.get_eol()
                        elif not token.is_eol_or_eof():
                            raise dns.exception.SyntaxError("bad origin in $INCLUDE")
                        else:
                            new_origin = self.current_origin
                        self.saved_state.append(
                            (
                                self.tok,
                                self.current_origin,
                                self.last_name,
                                self.current_file,
                                self.last_ttl,
                                self.last_ttl_known,
                                self.default_ttl,
                                self.default_ttl_known,
                            )
                        )
                        self.current_file = open(filename, "r")
                        self.tok = dns.tokenizer.Tokenizer(self.current_file, filename)
                        self.current_origin = new_origin
                    elif c == "$GENERATE":
                        self._generate_line()
                    else:
                        raise dns.exception.SyntaxError(
                            f"Unknown zone file directive '{c}'"
                        )
                    continue
                self.tok.unget(token)
                self._rr_line()
        except dns.exception.SyntaxError as detail:
            (filename, line_number) = self.tok.where()
            if detail is None:
                detail = "syntax error"
            ex = dns.exception.SyntaxError(
                "%s:%d: %s" % (filename, line_number, detail)
            )
            tb = sys.exc_info()[2]
            raise ex.with_traceback(tb) from None


class RRsetsReaderTransaction(dns.transaction.Transaction):
    def __init__(self, manager, replacement, read_only):
        assert not read_only
        super().__init__(manager, replacement, read_only)
        self.rdatasets = {}

    def _get_rdataset(self, name, rdtype, covers):
        return self.rdatasets.get((name, rdtype, covers))

    def _get_node(self, name):
        rdatasets = []
        for (rdataset_name, _, _), rdataset in self.rdatasets.items():
            if name == rdataset_name:
                rdatasets.append(rdataset)
        if len(rdatasets) == 0:
            return None
        node = dns.node.Node()
        node.rdatasets = rdatasets
        return node

    def _put_rdataset(self, name, rdataset):
        self.rdatasets[(name, rdataset.rdtype, rdataset.covers)] = rdataset

    def _delete_name(self, name):
        # First remove any changes involving the name
        remove = []
        for key in self.rdatasets:
            if key[0] == name:
                remove.append(key)
        if len(remove) > 0:
            for key in remove:
                del self.rdatasets[key]

    def _delete_rdataset(self, name, rdtype, covers):
        try:
            del self.rdatasets[(name, rdtype, covers)]
        except KeyError:
            pass

    def _name_exists(self, name):
        for (n, _, _) in self.rdatasets:
            if n == name:
                return True
        return False

    def _changed(self):
        return len(self.rdatasets) > 0

    def _end_transaction(self, commit):
        if commit and self._changed():
            rrsets = []
            for (name, _, _), rdataset in self.rdatasets.items():
                rrset = dns.rrset.RRset(
                    name, rdataset.rdclass, rdataset.rdtype, rdataset.covers
                )
                rrset.update(rdataset)
                rrsets.append(rrset)
            self.manager.set_rrsets(rrsets)

    def _set_origin(self, origin):
        pass

    def _iterate_rdatasets(self):
        raise NotImplementedError  # pragma: no cover


class RRSetsReaderManager(dns.transaction.TransactionManager):
    def __init__(
        self, origin=dns.name.root, relativize=False, rdclass=dns.rdataclass.IN
    ):
        self.origin = origin
        self.relativize = relativize
        self.rdclass = rdclass
        self.rrsets = []

    def reader(self):  # pragma: no cover
        raise NotImplementedError

    def writer(self, replacement=False):
        assert replacement is True
        return RRsetsReaderTransaction(self, True, False)

    def get_class(self):
        return self.rdclass

    def origin_information(self):
        if self.relativize:
            effective = dns.name.empty
        else:
            effective = self.origin
        return (self.origin, self.relativize, effective)

    def set_rrsets(self, rrsets):
        self.rrsets = rrsets


def read_rrsets(
    text: Any,
    name: Optional[Union[dns.name.Name, str]] = None,
    ttl: Optional[int] = None,
    rdclass: Optional[Union[dns.rdataclass.RdataClass, str]] = dns.rdataclass.IN,
    default_rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
    rdtype: Optional[Union[dns.rdatatype.RdataType, str]] = None,
    default_ttl: Optional[Union[int, str]] = None,
    idna_codec: Optional[dns.name.IDNACodec] = None,
    origin: Optional[Union[dns.name.Name, str]] = dns.name.root,
    relativize: bool = False,
) -> List[dns.rrset.RRset]:
    """Read one or more rrsets from the specified text, possibly subject
    to restrictions.

    *text*, a file object or a string, is the input to process.

    *name*, a string, ``dns.name.Name``, or ``None``, is the owner name of
    the rrset.  If not ``None``, then the owner name is "forced", and the
    input must not specify an owner name.  If ``None``, then any owner names
    are allowed and must be present in the input.

    *ttl*, an ``int``, string, or None.  If not ``None``, the the TTL is
    forced to be the specified value and the input must not specify a TTL.
    If ``None``, then a TTL may be specified in the input.  If it is not
    specified, then the *default_ttl* will be used.

    *rdclass*, a ``dns.rdataclass.RdataClass``, string, or ``None``.  If
    not ``None``, then the class is forced to the specified value, and the
    input must not specify a class.  If ``None``, then the input may specify
    a class that matches *default_rdclass*.  Note that it is not possible to
    return rrsets with differing classes; specifying ``None`` for the class
    simply allows the user to optionally type a class as that may be convenient
    when cutting and pasting.

    *default_rdclass*, a ``dns.rdataclass.RdataClass`` or string.  The class
    of the returned rrsets.

    *rdtype*, a ``dns.rdatatype.RdataType``, string, or ``None``.  If not
    ``None``, then the type is forced to the specified value, and the
    input must not specify a type.  If ``None``, then a type must be present
    for each RR.

    *default_ttl*, an ``int``, string, or ``None``.  If not ``None``, then if
    the TTL is not forced and is not specified, then this value will be used.
    if ``None``, then if the TTL is not forced an error will occur if the TTL
    is not specified.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.  Note that codecs only apply to the owner name; dnspython does
    not do IDNA for names in rdata, as there is no IDNA zonefile format.

    *origin*, a string, ``dns.name.Name``, or ``None``, is the origin for any
    relative names in the input, and also the origin to relativize to if
    *relativize* is ``True``.

    *relativize*, a bool.  If ``True``, names are relativized to the *origin*;
    if ``False`` then any relative names in the input are made absolute by
    appending the *origin*.
    """
    if isinstance(origin, str):
        origin = dns.name.from_text(origin, dns.name.root, idna_codec)
    if isinstance(name, str):
        name = dns.name.from_text(name, origin, idna_codec)
    if isinstance(ttl, str):
        ttl = dns.ttl.from_text(ttl)
    if isinstance(default_ttl, str):
        default_ttl = dns.ttl.from_text(default_ttl)
    if rdclass is not None:
        the_rdclass = dns.rdataclass.RdataClass.make(rdclass)
    else:
        the_rdclass = None
    the_default_rdclass = dns.rdataclass.RdataClass.make(default_rdclass)
    if rdtype is not None:
        the_rdtype = dns.rdatatype.RdataType.make(rdtype)
    else:
        the_rdtype = None
    manager = RRSetsReaderManager(origin, relativize, default_rdclass)
    with manager.writer(True) as txn:
        tok = dns.tokenizer.Tokenizer(text, "<input>", idna_codec=idna_codec)
        reader = Reader(
            tok,
            the_default_rdclass,
            txn,
            allow_directives=False,
            force_name=name,
            force_ttl=ttl,
            force_rdclass=the_rdclass,
            force_rdtype=the_rdtype,
            default_ttl=default_ttl,
        )
        reader.read()
    return manager.rrsets
