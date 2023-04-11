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

"""DNS Messages"""

from typing import Any, Dict, List, Optional, Tuple, Union

import contextlib
import io
import time

import dns.wire
import dns.edns
import dns.enum
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.ttl
import dns.tsig
import dns.rdtypes.ANY.OPT
import dns.rdtypes.ANY.TSIG


class ShortHeader(dns.exception.FormError):
    """The DNS packet passed to from_wire() is too short."""


class TrailingJunk(dns.exception.FormError):
    """The DNS packet passed to from_wire() has extra junk at the end of it."""


class UnknownHeaderField(dns.exception.DNSException):
    """The header field name was not recognized when converting from text
    into a message."""


class BadEDNS(dns.exception.FormError):
    """An OPT record occurred somewhere other than
    the additional data section."""


class BadTSIG(dns.exception.FormError):
    """A TSIG record occurred somewhere other than the end of
    the additional data section."""


class UnknownTSIGKey(dns.exception.DNSException):
    """A TSIG with an unknown key was received."""


class Truncated(dns.exception.DNSException):
    """The truncated flag is set."""

    supp_kwargs = {"message"}

    # We do this as otherwise mypy complains about unexpected keyword argument
    # idna_exception
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def message(self):
        """As much of the message as could be processed.

        Returns a ``dns.message.Message``.
        """
        return self.kwargs["message"]


class NotQueryResponse(dns.exception.DNSException):
    """Message is not a response to a query."""


class ChainTooLong(dns.exception.DNSException):
    """The CNAME chain is too long."""


class AnswerForNXDOMAIN(dns.exception.DNSException):
    """The rcode is NXDOMAIN but an answer was found."""


class NoPreviousName(dns.exception.SyntaxError):
    """No previous name was known."""


class MessageSection(dns.enum.IntEnum):
    """Message sections"""

    QUESTION = 0
    ANSWER = 1
    AUTHORITY = 2
    ADDITIONAL = 3

    @classmethod
    def _maximum(cls):
        return 3


class MessageError:
    def __init__(self, exception: Exception, offset: int):
        self.exception = exception
        self.offset = offset


DEFAULT_EDNS_PAYLOAD = 1232
MAX_CHAIN = 16

IndexKeyType = Tuple[
    int,
    dns.name.Name,
    dns.rdataclass.RdataClass,
    dns.rdatatype.RdataType,
    Optional[dns.rdatatype.RdataType],
    Optional[dns.rdataclass.RdataClass],
]
IndexType = Dict[IndexKeyType, dns.rrset.RRset]
SectionType = Union[int, List[dns.rrset.RRset]]


class Message:
    """A DNS message."""

    _section_enum = MessageSection

    def __init__(self, id: Optional[int] = None):
        if id is None:
            self.id = dns.entropy.random_16()
        else:
            self.id = id
        self.flags = 0
        self.sections: List[List[dns.rrset.RRset]] = [[], [], [], []]
        self.opt: Optional[dns.rrset.RRset] = None
        self.request_payload = 0
        self.pad = 0
        self.keyring: Any = None
        self.tsig: Optional[dns.rrset.RRset] = None
        self.request_mac = b""
        self.xfr = False
        self.origin: Optional[dns.name.Name] = None
        self.tsig_ctx: Optional[Any] = None
        self.index: IndexType = {}
        self.errors: List[MessageError] = []
        self.time = 0.0

    @property
    def question(self) -> List[dns.rrset.RRset]:
        """The question section."""
        return self.sections[0]

    @question.setter
    def question(self, v):
        self.sections[0] = v

    @property
    def answer(self) -> List[dns.rrset.RRset]:
        """The answer section."""
        return self.sections[1]

    @answer.setter
    def answer(self, v):
        self.sections[1] = v

    @property
    def authority(self) -> List[dns.rrset.RRset]:
        """The authority section."""
        return self.sections[2]

    @authority.setter
    def authority(self, v):
        self.sections[2] = v

    @property
    def additional(self) -> List[dns.rrset.RRset]:
        """The additional data section."""
        return self.sections[3]

    @additional.setter
    def additional(self, v):
        self.sections[3] = v

    def __repr__(self):
        return "<DNS message, ID " + repr(self.id) + ">"

    def __str__(self):
        return self.to_text()

    def to_text(
        self,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        **kw: Dict[str, Any],
    ) -> str:
        """Convert the message to text.

        The *origin*, *relativize*, and any other keyword
        arguments are passed to the RRset ``to_wire()`` method.

        Returns a ``str``.
        """

        s = io.StringIO()
        s.write("id %d\n" % self.id)
        s.write("opcode %s\n" % dns.opcode.to_text(self.opcode()))
        s.write("rcode %s\n" % dns.rcode.to_text(self.rcode()))
        s.write("flags %s\n" % dns.flags.to_text(self.flags))
        if self.edns >= 0:
            s.write("edns %s\n" % self.edns)
            if self.ednsflags != 0:
                s.write("eflags %s\n" % dns.flags.edns_to_text(self.ednsflags))
            s.write("payload %d\n" % self.payload)
        for opt in self.options:
            s.write("option %s\n" % opt.to_text())
        for (name, which) in self._section_enum.__members__.items():
            s.write(f";{name}\n")
            for rrset in self.section_from_number(which):
                s.write(rrset.to_text(origin, relativize, **kw))
                s.write("\n")
        #
        # We strip off the final \n so the caller can print the result without
        # doing weird things to get around eccentricities in Python print
        # formatting
        #
        return s.getvalue()[:-1]

    def __eq__(self, other):
        """Two messages are equal if they have the same content in the
        header, question, answer, and authority sections.

        Returns a ``bool``.
        """

        if not isinstance(other, Message):
            return False
        if self.id != other.id:
            return False
        if self.flags != other.flags:
            return False
        for i, section in enumerate(self.sections):
            other_section = other.sections[i]
            for n in section:
                if n not in other_section:
                    return False
            for n in other_section:
                if n not in section:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_response(self, other: "Message") -> bool:
        """Is *other*, also a ``dns.message.Message``, a response to this
        message?

        Returns a ``bool``.
        """

        if (
            other.flags & dns.flags.QR == 0
            or self.id != other.id
            or dns.opcode.from_flags(self.flags) != dns.opcode.from_flags(other.flags)
        ):
            return False
        if other.rcode() in {
            dns.rcode.FORMERR,
            dns.rcode.SERVFAIL,
            dns.rcode.NOTIMP,
            dns.rcode.REFUSED,
        }:
            # We don't check the question section in these cases if
            # the other question section is empty, even though they
            # still really ought to have a question section.
            if len(other.question) == 0:
                return True
        if dns.opcode.is_update(self.flags):
            # This is assuming the "sender doesn't include anything
            # from the update", but we don't care to check the other
            # case, which is that all the sections are returned and
            # identical.
            return True
        for n in self.question:
            if n not in other.question:
                return False
        for n in other.question:
            if n not in self.question:
                return False
        return True

    def section_number(self, section: List[dns.rrset.RRset]) -> int:
        """Return the "section number" of the specified section for use
        in indexing.

        *section* is one of the section attributes of this message.

        Raises ``ValueError`` if the section isn't known.

        Returns an ``int``.
        """

        for i, our_section in enumerate(self.sections):
            if section is our_section:
                return self._section_enum(i)
        raise ValueError("unknown section")

    def section_from_number(self, number: int) -> List[dns.rrset.RRset]:
        """Return the section list associated with the specified section
        number.

        *number* is a section number `int` or the text form of a section
        name.

        Raises ``ValueError`` if the section isn't known.

        Returns a ``list``.
        """

        section = self._section_enum.make(number)
        return self.sections[section]

    def find_rrset(
        self,
        section: SectionType,
        name: dns.name.Name,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        deleting: Optional[dns.rdataclass.RdataClass] = None,
        create: bool = False,
        force_unique: bool = False,
    ) -> dns.rrset.RRset:
        """Find the RRset with the given attributes in the specified section.

        *section*, an ``int`` section number, or one of the section
        attributes of this message.  This specifies the
        the section of the message to search.  For example::

            my_message.find_rrset(my_message.answer, name, rdclass, rdtype)
            my_message.find_rrset(dns.message.ANSWER, name, rdclass, rdtype)

        *name*, a ``dns.name.Name``, the name of the RRset.

        *rdclass*, an ``int``, the class of the RRset.

        *rdtype*, an ``int``, the type of the RRset.

        *covers*, an ``int`` or ``None``, the covers value of the RRset.
        The default is ``None``.

        *deleting*, an ``int`` or ``None``, the deleting value of the RRset.
        The default is ``None``.

        *create*, a ``bool``.  If ``True``, create the RRset if it is not found.
        The created RRset is appended to *section*.

        *force_unique*, a ``bool``.  If ``True`` and *create* is also ``True``,
        create a new RRset regardless of whether a matching RRset exists
        already.  The default is ``False``.  This is useful when creating
        DDNS Update messages, as order matters for them.

        Raises ``KeyError`` if the RRset was not found and create was
        ``False``.

        Returns a ``dns.rrset.RRset object``.
        """

        if isinstance(section, int):
            section_number = section
            the_section = self.section_from_number(section_number)
        else:
            section_number = self.section_number(section)
            the_section = section
        key = (section_number, name, rdclass, rdtype, covers, deleting)
        if not force_unique:
            if self.index is not None:
                rrset = self.index.get(key)
                if rrset is not None:
                    return rrset
            else:
                for rrset in the_section:
                    if rrset.full_match(name, rdclass, rdtype, covers, deleting):
                        return rrset
        if not create:
            raise KeyError
        rrset = dns.rrset.RRset(name, rdclass, rdtype, covers, deleting)
        the_section.append(rrset)
        if self.index is not None:
            self.index[key] = rrset
        return rrset

    def get_rrset(
        self,
        section: SectionType,
        name: dns.name.Name,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        covers: dns.rdatatype.RdataType = dns.rdatatype.NONE,
        deleting: Optional[dns.rdataclass.RdataClass] = None,
        create: bool = False,
        force_unique: bool = False,
    ) -> Optional[dns.rrset.RRset]:
        """Get the RRset with the given attributes in the specified section.

        If the RRset is not found, None is returned.

        *section*, an ``int`` section number, or one of the section
        attributes of this message.  This specifies the
        the section of the message to search.  For example::

            my_message.get_rrset(my_message.answer, name, rdclass, rdtype)
            my_message.get_rrset(dns.message.ANSWER, name, rdclass, rdtype)

        *name*, a ``dns.name.Name``, the name of the RRset.

        *rdclass*, an ``int``, the class of the RRset.

        *rdtype*, an ``int``, the type of the RRset.

        *covers*, an ``int`` or ``None``, the covers value of the RRset.
        The default is ``None``.

        *deleting*, an ``int`` or ``None``, the deleting value of the RRset.
        The default is ``None``.

        *create*, a ``bool``.  If ``True``, create the RRset if it is not found.
        The created RRset is appended to *section*.

        *force_unique*, a ``bool``.  If ``True`` and *create* is also ``True``,
        create a new RRset regardless of whether a matching RRset exists
        already.  The default is ``False``.  This is useful when creating
        DDNS Update messages, as order matters for them.

        Returns a ``dns.rrset.RRset object`` or ``None``.
        """

        try:
            rrset = self.find_rrset(
                section, name, rdclass, rdtype, covers, deleting, create, force_unique
            )
        except KeyError:
            rrset = None
        return rrset

    def _compute_opt_reserve(self) -> int:
        """Compute the size required for the OPT RR, padding excluded"""
        if not self.opt:
            return 0
        # 1 byte for the root name, 10 for the standard RR fields
        size = 11
        # This would be more efficient if options had a size() method, but we won't
        # worry about that for now.  We also don't worry if there is an existing padding
        # option, as it is unlikely and probably harmless, as the worst case is that we
        # may add another, and this seems to be legal.
        for option in self.opt[0].options:
            wire = option.to_wire()
            # We add 4 here to account for the option type and length
            size += len(wire) + 4
        if self.pad:
            # Padding will be added, so again add the option type and length.
            size += 4
        return size

    def _compute_tsig_reserve(self) -> int:
        """Compute the size required for the TSIG RR"""
        # This would be more efficient if TSIGs had a size method, but we won't
        # worry about for now.  Also, we can't really cope with the potential
        # compressibility of the TSIG owner name, so we estimate with the uncompressed
        # size.  We will disable compression when TSIG and padding are both is active
        # so that the padding comes out right.
        if not self.tsig:
            return 0
        f = io.BytesIO()
        self.tsig.to_wire(f)
        return len(f.getvalue())

    def to_wire(
        self,
        origin: Optional[dns.name.Name] = None,
        max_size: int = 0,
        multi: bool = False,
        tsig_ctx: Optional[Any] = None,
        **kw: Dict[str, Any],
    ) -> bytes:
        """Return a string containing the message in DNS compressed wire
        format.

        Additional keyword arguments are passed to the RRset ``to_wire()``
        method.

        *origin*, a ``dns.name.Name`` or ``None``, the origin to be appended
        to any relative names.  If ``None``, and the message has an origin
        attribute that is not ``None``, then it will be used.

        *max_size*, an ``int``, the maximum size of the wire format
        output; default is 0, which means "the message's request
        payload, if nonzero, or 65535".

        *multi*, a ``bool``, should be set to ``True`` if this message is
        part of a multiple message sequence.

        *tsig_ctx*, a ``dns.tsig.HMACTSig`` or ``dns.tsig.GSSTSig`` object, the
        ongoing TSIG context, used when signing zone transfers.

        Raises ``dns.exception.TooBig`` if *max_size* was exceeded.

        Returns a ``bytes``.
        """

        if origin is None and self.origin is not None:
            origin = self.origin
        if max_size == 0:
            if self.request_payload != 0:
                max_size = self.request_payload
            else:
                max_size = 65535
        if max_size < 512:
            max_size = 512
        elif max_size > 65535:
            max_size = 65535
        r = dns.renderer.Renderer(self.id, self.flags, max_size, origin)
        opt_reserve = self._compute_opt_reserve()
        r.reserve(opt_reserve)
        tsig_reserve = self._compute_tsig_reserve()
        r.reserve(tsig_reserve)
        for rrset in self.question:
            r.add_question(rrset.name, rrset.rdtype, rrset.rdclass)
        for rrset in self.answer:
            r.add_rrset(dns.renderer.ANSWER, rrset, **kw)
        for rrset in self.authority:
            r.add_rrset(dns.renderer.AUTHORITY, rrset, **kw)
        for rrset in self.additional:
            r.add_rrset(dns.renderer.ADDITIONAL, rrset, **kw)
        r.release_reserved()
        if self.opt is not None:
            r.add_opt(self.opt, self.pad, opt_reserve, tsig_reserve)
        r.write_header()
        if self.tsig is not None:
            (new_tsig, ctx) = dns.tsig.sign(
                r.get_wire(),
                self.keyring,
                self.tsig[0],
                int(time.time()),
                self.request_mac,
                tsig_ctx,
                multi,
            )
            self.tsig.clear()
            self.tsig.add(new_tsig)
            r.add_rrset(dns.renderer.ADDITIONAL, self.tsig)
            r.write_header()
            if multi:
                self.tsig_ctx = ctx
        return r.get_wire()

    @staticmethod
    def _make_tsig(
        keyname, algorithm, time_signed, fudge, mac, original_id, error, other
    ):
        tsig = dns.rdtypes.ANY.TSIG.TSIG(
            dns.rdataclass.ANY,
            dns.rdatatype.TSIG,
            algorithm,
            time_signed,
            fudge,
            mac,
            original_id,
            error,
            other,
        )
        return dns.rrset.from_rdata(keyname, 0, tsig)

    def use_tsig(
        self,
        keyring: Any,
        keyname: Optional[Union[dns.name.Name, str]] = None,
        fudge: int = 300,
        original_id: Optional[int] = None,
        tsig_error: int = 0,
        other_data: bytes = b"",
        algorithm: Union[dns.name.Name, str] = dns.tsig.default_algorithm,
    ) -> None:
        """When sending, a TSIG signature using the specified key
        should be added.

        *key*, a ``dns.tsig.Key`` is the key to use.  If a key is specified,
        the *keyring* and *algorithm* fields are not used.

        *keyring*, a ``dict``, ``callable`` or ``dns.tsig.Key``, is either
        the TSIG keyring or key to use.

        The format of a keyring dict is a mapping from TSIG key name, as
        ``dns.name.Name`` to ``dns.tsig.Key`` or a TSIG secret, a ``bytes``.
        If a ``dict`` *keyring* is specified but a *keyname* is not, the key
        used will be the first key in the *keyring*.  Note that the order of
        keys in a dictionary is not defined, so applications should supply a
        keyname when a ``dict`` keyring is used, unless they know the keyring
        contains only one key.  If a ``callable`` keyring is specified, the
        callable will be called with the message and the keyname, and is
        expected to return a key.

        *keyname*, a ``dns.name.Name``, ``str`` or ``None``, the name of
        this TSIG key to use; defaults to ``None``.  If *keyring* is a
        ``dict``, the key must be defined in it.  If *keyring* is a
        ``dns.tsig.Key``, this is ignored.

        *fudge*, an ``int``, the TSIG time fudge.

        *original_id*, an ``int``, the TSIG original id.  If ``None``,
        the message's id is used.

        *tsig_error*, an ``int``, the TSIG error code.

        *other_data*, a ``bytes``, the TSIG other data.

        *algorithm*, a ``dns.name.Name`` or ``str``, the TSIG algorithm to use.  This is
        only used if *keyring* is a ``dict``, and the key entry is a ``bytes``.
        """

        if isinstance(keyring, dns.tsig.Key):
            key = keyring
            keyname = key.name
        elif callable(keyring):
            key = keyring(self, keyname)
        else:
            if isinstance(keyname, str):
                keyname = dns.name.from_text(keyname)
            if keyname is None:
                keyname = next(iter(keyring))
            key = keyring[keyname]
            if isinstance(key, bytes):
                key = dns.tsig.Key(keyname, key, algorithm)
        self.keyring = key
        if original_id is None:
            original_id = self.id
        self.tsig = self._make_tsig(
            keyname,
            self.keyring.algorithm,
            0,
            fudge,
            b"\x00" * dns.tsig.mac_sizes[self.keyring.algorithm],
            original_id,
            tsig_error,
            other_data,
        )

    @property
    def keyname(self) -> Optional[dns.name.Name]:
        if self.tsig:
            return self.tsig.name
        else:
            return None

    @property
    def keyalgorithm(self) -> Optional[dns.name.Name]:
        if self.tsig:
            return self.tsig[0].algorithm
        else:
            return None

    @property
    def mac(self) -> Optional[bytes]:
        if self.tsig:
            return self.tsig[0].mac
        else:
            return None

    @property
    def tsig_error(self) -> Optional[int]:
        if self.tsig:
            return self.tsig[0].error
        else:
            return None

    @property
    def had_tsig(self) -> bool:
        return bool(self.tsig)

    @staticmethod
    def _make_opt(flags=0, payload=DEFAULT_EDNS_PAYLOAD, options=None):
        opt = dns.rdtypes.ANY.OPT.OPT(payload, dns.rdatatype.OPT, options or ())
        return dns.rrset.from_rdata(dns.name.root, int(flags), opt)

    def use_edns(
        self,
        edns: Optional[Union[int, bool]] = 0,
        ednsflags: int = 0,
        payload: int = DEFAULT_EDNS_PAYLOAD,
        request_payload: Optional[int] = None,
        options: Optional[List[dns.edns.Option]] = None,
        pad: int = 0,
    ) -> None:
        """Configure EDNS behavior.

        *edns*, an ``int``, is the EDNS level to use.  Specifying ``None``, ``False``,
        or ``-1`` means "do not use EDNS", and in this case the other parameters are
        ignored.  Specifying ``True`` is equivalent to specifying 0, i.e. "use EDNS0".

        *ednsflags*, an ``int``, the EDNS flag values.

        *payload*, an ``int``, is the EDNS sender's payload field, which is the maximum
        size of UDP datagram the sender can handle.  I.e. how big a response to this
        message can be.

        *request_payload*, an ``int``, is the EDNS payload size to use when sending this
        message.  If not specified, defaults to the value of *payload*.

        *options*, a list of ``dns.edns.Option`` objects or ``None``, the EDNS options.

        *pad*, a non-negative ``int``.  If 0, the default, do not pad; otherwise add
        padding bytes to make the message size a multiple of *pad*.  Note that if
        padding is non-zero, an EDNS PADDING option will always be added to the
        message.
        """

        if edns is None or edns is False:
            edns = -1
        elif edns is True:
            edns = 0
        if edns < 0:
            self.opt = None
            self.request_payload = 0
        else:
            # make sure the EDNS version in ednsflags agrees with edns
            ednsflags &= 0xFF00FFFF
            ednsflags |= edns << 16
            if options is None:
                options = []
            self.opt = self._make_opt(ednsflags, payload, options)
            if request_payload is None:
                request_payload = payload
            self.request_payload = request_payload
            self.pad = pad

    @property
    def edns(self) -> int:
        if self.opt:
            return (self.ednsflags & 0xFF0000) >> 16
        else:
            return -1

    @property
    def ednsflags(self) -> int:
        if self.opt:
            return self.opt.ttl
        else:
            return 0

    @ednsflags.setter
    def ednsflags(self, v):
        if self.opt:
            self.opt.ttl = v
        elif v:
            self.opt = self._make_opt(v)

    @property
    def payload(self) -> int:
        if self.opt:
            return self.opt[0].payload
        else:
            return 0

    @property
    def options(self) -> Tuple:
        if self.opt:
            return self.opt[0].options
        else:
            return ()

    def want_dnssec(self, wanted: bool = True) -> None:
        """Enable or disable 'DNSSEC desired' flag in requests.

        *wanted*, a ``bool``.  If ``True``, then DNSSEC data is
        desired in the response, EDNS is enabled if required, and then
        the DO bit is set.  If ``False``, the DO bit is cleared if
        EDNS is enabled.
        """

        if wanted:
            self.ednsflags |= dns.flags.DO
        elif self.opt:
            self.ednsflags &= ~dns.flags.DO

    def rcode(self) -> dns.rcode.Rcode:
        """Return the rcode.

        Returns a ``dns.rcode.Rcode``.
        """
        return dns.rcode.from_flags(int(self.flags), int(self.ednsflags))

    def set_rcode(self, rcode: dns.rcode.Rcode) -> None:
        """Set the rcode.

        *rcode*, a ``dns.rcode.Rcode``, is the rcode to set.
        """
        (value, evalue) = dns.rcode.to_flags(rcode)
        self.flags &= 0xFFF0
        self.flags |= value
        self.ednsflags &= 0x00FFFFFF
        self.ednsflags |= evalue

    def opcode(self) -> dns.opcode.Opcode:
        """Return the opcode.

        Returns a ``dns.opcode.Opcode``.
        """
        return dns.opcode.from_flags(int(self.flags))

    def set_opcode(self, opcode: dns.opcode.Opcode) -> None:
        """Set the opcode.

        *opcode*, a ``dns.opcode.Opcode``, is the opcode to set.
        """
        self.flags &= 0x87FF
        self.flags |= dns.opcode.to_flags(opcode)

    def _get_one_rr_per_rrset(self, value):
        # What the caller picked is fine.
        return value

    # pylint: disable=unused-argument

    def _parse_rr_header(self, section, name, rdclass, rdtype):
        return (rdclass, rdtype, None, False)

    # pylint: enable=unused-argument

    def _parse_special_rr_header(self, section, count, position, name, rdclass, rdtype):
        if rdtype == dns.rdatatype.OPT:
            if (
                section != MessageSection.ADDITIONAL
                or self.opt
                or name != dns.name.root
            ):
                raise BadEDNS
        elif rdtype == dns.rdatatype.TSIG:
            if (
                section != MessageSection.ADDITIONAL
                or rdclass != dns.rdatatype.ANY
                or position != count - 1
            ):
                raise BadTSIG
        return (rdclass, rdtype, None, False)


class ChainingResult:
    """The result of a call to dns.message.QueryMessage.resolve_chaining().

    The ``answer`` attribute is the answer RRSet, or ``None`` if it doesn't
    exist.

    The ``canonical_name`` attribute is the canonical name after all
    chaining has been applied (this is the same name as ``rrset.name`` in cases
    where rrset is not ``None``).

    The ``minimum_ttl`` attribute is the minimum TTL, i.e. the TTL to
    use if caching the data.  It is the smallest of all the CNAME TTLs
    and either the answer TTL if it exists or the SOA TTL and SOA
    minimum values for negative answers.

    The ``cnames`` attribute is a list of all the CNAME RRSets followed to
    get to the canonical name.
    """

    def __init__(
        self,
        canonical_name: dns.name.Name,
        answer: Optional[dns.rrset.RRset],
        minimum_ttl: int,
        cnames: List[dns.rrset.RRset],
    ):
        self.canonical_name = canonical_name
        self.answer = answer
        self.minimum_ttl = minimum_ttl
        self.cnames = cnames


class QueryMessage(Message):
    def resolve_chaining(self) -> ChainingResult:
        """Follow the CNAME chain in the response to determine the answer
        RRset.

        Raises ``dns.message.NotQueryResponse`` if the message is not
        a response.

        Raises ``dns.message.ChainTooLong`` if the CNAME chain is too long.

        Raises ``dns.message.AnswerForNXDOMAIN`` if the rcode is NXDOMAIN
        but an answer was found.

        Raises ``dns.exception.FormError`` if the question count is not 1.

        Returns a ChainingResult object.
        """
        if self.flags & dns.flags.QR == 0:
            raise NotQueryResponse
        if len(self.question) != 1:
            raise dns.exception.FormError
        question = self.question[0]
        qname = question.name
        min_ttl = dns.ttl.MAX_TTL
        answer = None
        count = 0
        cnames = []
        while count < MAX_CHAIN:
            try:
                answer = self.find_rrset(
                    self.answer, qname, question.rdclass, question.rdtype
                )
                min_ttl = min(min_ttl, answer.ttl)
                break
            except KeyError:
                if question.rdtype != dns.rdatatype.CNAME:
                    try:
                        crrset = self.find_rrset(
                            self.answer, qname, question.rdclass, dns.rdatatype.CNAME
                        )
                        cnames.append(crrset)
                        min_ttl = min(min_ttl, crrset.ttl)
                        for rd in crrset:
                            qname = rd.target
                            break
                        count += 1
                        continue
                    except KeyError:
                        # Exit the chaining loop
                        break
                else:
                    # Exit the chaining loop
                    break
        if count >= MAX_CHAIN:
            raise ChainTooLong
        if self.rcode() == dns.rcode.NXDOMAIN and answer is not None:
            raise AnswerForNXDOMAIN
        if answer is None:
            # Further minimize the TTL with NCACHE.
            auname = qname
            while True:
                # Look for an SOA RR whose owner name is a superdomain
                # of qname.
                try:
                    srrset = self.find_rrset(
                        self.authority, auname, question.rdclass, dns.rdatatype.SOA
                    )
                    min_ttl = min(min_ttl, srrset.ttl, srrset[0].minimum)
                    break
                except KeyError:
                    try:
                        auname = auname.parent()
                    except dns.name.NoParent:
                        break
        return ChainingResult(qname, answer, min_ttl, cnames)

    def canonical_name(self) -> dns.name.Name:
        """Return the canonical name of the first name in the question
        section.

        Raises ``dns.message.NotQueryResponse`` if the message is not
        a response.

        Raises ``dns.message.ChainTooLong`` if the CNAME chain is too long.

        Raises ``dns.message.AnswerForNXDOMAIN`` if the rcode is NXDOMAIN
        but an answer was found.

        Raises ``dns.exception.FormError`` if the question count is not 1.
        """
        return self.resolve_chaining().canonical_name


def _maybe_import_update():
    # We avoid circular imports by doing this here.  We do it in another
    # function as doing it in _message_factory_from_opcode() makes "dns"
    # a local symbol, and the first line fails :)

    # pylint: disable=redefined-outer-name,import-outside-toplevel,unused-import
    import dns.update  # noqa: F401


def _message_factory_from_opcode(opcode):
    if opcode == dns.opcode.QUERY:
        return QueryMessage
    elif opcode == dns.opcode.UPDATE:
        _maybe_import_update()
        return dns.update.UpdateMessage
    else:
        return Message


class _WireReader:

    """Wire format reader.

    parser: the binary parser
    message: The message object being built
    initialize_message: Callback to set message parsing options
    question_only: Are we only reading the question?
    one_rr_per_rrset: Put each RR into its own RRset?
    keyring: TSIG keyring
    ignore_trailing: Ignore trailing junk at end of request?
    multi: Is this message part of a multi-message sequence?
    DNS dynamic updates.
    continue_on_error: try to extract as much information as possible from
    the message, accumulating MessageErrors in the *errors* attribute instead of
    raising them.
    """

    def __init__(
        self,
        wire,
        initialize_message,
        question_only=False,
        one_rr_per_rrset=False,
        ignore_trailing=False,
        keyring=None,
        multi=False,
        continue_on_error=False,
    ):
        self.parser = dns.wire.Parser(wire)
        self.message = None
        self.initialize_message = initialize_message
        self.question_only = question_only
        self.one_rr_per_rrset = one_rr_per_rrset
        self.ignore_trailing = ignore_trailing
        self.keyring = keyring
        self.multi = multi
        self.continue_on_error = continue_on_error
        self.errors = []

    def _get_question(self, section_number, qcount):
        """Read the next *qcount* records from the wire data and add them to
        the question section.
        """
        assert self.message is not None
        section = self.message.sections[section_number]
        for _ in range(qcount):
            qname = self.parser.get_name(self.message.origin)
            (rdtype, rdclass) = self.parser.get_struct("!HH")
            (rdclass, rdtype, _, _) = self.message._parse_rr_header(
                section_number, qname, rdclass, rdtype
            )
            self.message.find_rrset(
                section, qname, rdclass, rdtype, create=True, force_unique=True
            )

    def _add_error(self, e):
        self.errors.append(MessageError(e, self.parser.current))

    def _get_section(self, section_number, count):
        """Read the next I{count} records from the wire data and add them to
        the specified section.

        section_number: the section of the message to which to add records
        count: the number of records to read
        """
        assert self.message is not None
        section = self.message.sections[section_number]
        force_unique = self.one_rr_per_rrset
        for i in range(count):
            rr_start = self.parser.current
            absolute_name = self.parser.get_name()
            if self.message.origin is not None:
                name = absolute_name.relativize(self.message.origin)
            else:
                name = absolute_name
            (rdtype, rdclass, ttl, rdlen) = self.parser.get_struct("!HHIH")
            if rdtype in (dns.rdatatype.OPT, dns.rdatatype.TSIG):
                (
                    rdclass,
                    rdtype,
                    deleting,
                    empty,
                ) = self.message._parse_special_rr_header(
                    section_number, count, i, name, rdclass, rdtype
                )
            else:
                (rdclass, rdtype, deleting, empty) = self.message._parse_rr_header(
                    section_number, name, rdclass, rdtype
                )
            rdata_start = self.parser.current
            try:
                if empty:
                    if rdlen > 0:
                        raise dns.exception.FormError
                    rd = None
                    covers = dns.rdatatype.NONE
                else:
                    with self.parser.restrict_to(rdlen):
                        rd = dns.rdata.from_wire_parser(
                            rdclass, rdtype, self.parser, self.message.origin
                        )
                    covers = rd.covers()
                if self.message.xfr and rdtype == dns.rdatatype.SOA:
                    force_unique = True
                if rdtype == dns.rdatatype.OPT:
                    self.message.opt = dns.rrset.from_rdata(name, ttl, rd)
                elif rdtype == dns.rdatatype.TSIG:
                    if self.keyring is None:
                        raise UnknownTSIGKey("got signed message without keyring")
                    if isinstance(self.keyring, dict):
                        key = self.keyring.get(absolute_name)
                        if isinstance(key, bytes):
                            key = dns.tsig.Key(absolute_name, key, rd.algorithm)
                    elif callable(self.keyring):
                        key = self.keyring(self.message, absolute_name)
                    else:
                        key = self.keyring
                    if key is None:
                        raise UnknownTSIGKey("key '%s' unknown" % name)
                    self.message.keyring = key
                    self.message.tsig_ctx = dns.tsig.validate(
                        self.parser.wire,
                        key,
                        absolute_name,
                        rd,
                        int(time.time()),
                        self.message.request_mac,
                        rr_start,
                        self.message.tsig_ctx,
                        self.multi,
                    )
                    self.message.tsig = dns.rrset.from_rdata(absolute_name, 0, rd)
                else:
                    rrset = self.message.find_rrset(
                        section,
                        name,
                        rdclass,
                        rdtype,
                        covers,
                        deleting,
                        True,
                        force_unique,
                    )
                    if rd is not None:
                        if ttl > 0x7FFFFFFF:
                            ttl = 0
                        rrset.add(rd, ttl)
            except Exception as e:
                if self.continue_on_error:
                    self._add_error(e)
                    self.parser.seek(rdata_start + rdlen)
                else:
                    raise

    def read(self):
        """Read a wire format DNS message and build a dns.message.Message
        object."""

        if self.parser.remaining() < 12:
            raise ShortHeader
        (id, flags, qcount, ancount, aucount, adcount) = self.parser.get_struct(
            "!HHHHHH"
        )
        factory = _message_factory_from_opcode(dns.opcode.from_flags(flags))
        self.message = factory(id=id)
        self.message.flags = dns.flags.Flag(flags)
        self.initialize_message(self.message)
        self.one_rr_per_rrset = self.message._get_one_rr_per_rrset(
            self.one_rr_per_rrset
        )
        try:
            self._get_question(MessageSection.QUESTION, qcount)
            if self.question_only:
                return self.message
            self._get_section(MessageSection.ANSWER, ancount)
            self._get_section(MessageSection.AUTHORITY, aucount)
            self._get_section(MessageSection.ADDITIONAL, adcount)
            if not self.ignore_trailing and self.parser.remaining() != 0:
                raise TrailingJunk
            if self.multi and self.message.tsig_ctx and not self.message.had_tsig:
                self.message.tsig_ctx.update(self.parser.wire)
        except Exception as e:
            if self.continue_on_error:
                self._add_error(e)
            else:
                raise
        return self.message


def from_wire(
    wire: bytes,
    keyring: Optional[Any] = None,
    request_mac: Optional[bytes] = b"",
    xfr: bool = False,
    origin: Optional[dns.name.Name] = None,
    tsig_ctx: Optional[Union[dns.tsig.HMACTSig, dns.tsig.GSSTSig]] = None,
    multi: bool = False,
    question_only: bool = False,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    raise_on_truncation: bool = False,
    continue_on_error: bool = False,
) -> Message:
    """Convert a DNS wire format message into a message object.

    *keyring*, a ``dns.tsig.Key`` or ``dict``, the key or keyring to use if the message
    is signed.

    *request_mac*, a ``bytes`` or ``None``.  If the message is a response to a
    TSIG-signed request, *request_mac* should be set to the MAC of that request.

    *xfr*, a ``bool``, should be set to ``True`` if this message is part of a zone
    transfer.

    *origin*, a ``dns.name.Name`` or ``None``.  If the message is part of a zone
    transfer, *origin* should be the origin name of the zone.  If not ``None``, names
    will be relativized to the origin.

    *tsig_ctx*, a ``dns.tsig.HMACTSig`` or ``dns.tsig.GSSTSig`` object, the ongoing TSIG
    context, used when validating zone transfers.

    *multi*, a ``bool``, should be set to ``True`` if this message is part of a multiple
    message sequence.

    *question_only*, a ``bool``.  If ``True``, read only up to the end of the question
    section.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing junk at end of the
    message.

    *raise_on_truncation*, a ``bool``.  If ``True``, raise an exception if the TC bit is
    set.

    *continue_on_error*, a ``bool``.  If ``True``, try to continue parsing even if
    errors occur.  Erroneous rdata will be ignored.  Errors will be accumulated as a
    list of MessageError objects in the message's ``errors`` attribute.  This option is
    recommended only for DNS analysis tools, or for use in a server as part of an error
    handling path.  The default is ``False``.

    Raises ``dns.message.ShortHeader`` if the message is less than 12 octets long.

    Raises ``dns.message.TrailingJunk`` if there were octets in the message past the end
    of the proper DNS message, and *ignore_trailing* is ``False``.

    Raises ``dns.message.BadEDNS`` if an OPT record was in the wrong section, or
    occurred more than once.

    Raises ``dns.message.BadTSIG`` if a TSIG record was not the last record of the
    additional data section.

    Raises ``dns.message.Truncated`` if the TC flag is set and *raise_on_truncation* is
    ``True``.

    Returns a ``dns.message.Message``.
    """

    # We permit None for request_mac solely for backwards compatibility
    if request_mac is None:
        request_mac = b""

    def initialize_message(message):
        message.request_mac = request_mac
        message.xfr = xfr
        message.origin = origin
        message.tsig_ctx = tsig_ctx

    reader = _WireReader(
        wire,
        initialize_message,
        question_only,
        one_rr_per_rrset,
        ignore_trailing,
        keyring,
        multi,
        continue_on_error,
    )
    try:
        m = reader.read()
    except dns.exception.FormError:
        if (
            reader.message
            and (reader.message.flags & dns.flags.TC)
            and raise_on_truncation
        ):
            raise Truncated(message=reader.message)
        else:
            raise
    # Reading a truncated message might not have any errors, so we
    # have to do this check here too.
    if m.flags & dns.flags.TC and raise_on_truncation:
        raise Truncated(message=m)
    if continue_on_error:
        m.errors = reader.errors

    return m


class _TextReader:

    """Text format reader.

    tok: the tokenizer.
    message: The message object being built.
    DNS dynamic updates.
    last_name: The most recently read name when building a message object.
    one_rr_per_rrset: Put each RR into its own RRset?
    origin: The origin for relative names
    relativize: relativize names?
    relativize_to: the origin to relativize to.
    """

    def __init__(
        self,
        text,
        idna_codec,
        one_rr_per_rrset=False,
        origin=None,
        relativize=True,
        relativize_to=None,
    ):
        self.message = None
        self.tok = dns.tokenizer.Tokenizer(text, idna_codec=idna_codec)
        self.last_name = None
        self.one_rr_per_rrset = one_rr_per_rrset
        self.origin = origin
        self.relativize = relativize
        self.relativize_to = relativize_to
        self.id = None
        self.edns = -1
        self.ednsflags = 0
        self.payload = DEFAULT_EDNS_PAYLOAD
        self.rcode = None
        self.opcode = dns.opcode.QUERY
        self.flags = 0

    def _header_line(self, _):
        """Process one line from the text format header section."""

        token = self.tok.get()
        what = token.value
        if what == "id":
            self.id = self.tok.get_int()
        elif what == "flags":
            while True:
                token = self.tok.get()
                if not token.is_identifier():
                    self.tok.unget(token)
                    break
                self.flags = self.flags | dns.flags.from_text(token.value)
        elif what == "edns":
            self.edns = self.tok.get_int()
            self.ednsflags = self.ednsflags | (self.edns << 16)
        elif what == "eflags":
            if self.edns < 0:
                self.edns = 0
            while True:
                token = self.tok.get()
                if not token.is_identifier():
                    self.tok.unget(token)
                    break
                self.ednsflags = self.ednsflags | dns.flags.edns_from_text(token.value)
        elif what == "payload":
            self.payload = self.tok.get_int()
            if self.edns < 0:
                self.edns = 0
        elif what == "opcode":
            text = self.tok.get_string()
            self.opcode = dns.opcode.from_text(text)
            self.flags = self.flags | dns.opcode.to_flags(self.opcode)
        elif what == "rcode":
            text = self.tok.get_string()
            self.rcode = dns.rcode.from_text(text)
        else:
            raise UnknownHeaderField
        self.tok.get_eol()

    def _question_line(self, section_number):
        """Process one line from the text format question section."""

        section = self.message.sections[section_number]
        token = self.tok.get(want_leading=True)
        if not token.is_whitespace():
            self.last_name = self.tok.as_name(
                token, self.message.origin, self.relativize, self.relativize_to
            )
        name = self.last_name
        if name is None:
            raise NoPreviousName
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
        # Class
        try:
            rdclass = dns.rdataclass.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.exception.SyntaxError:
            raise dns.exception.SyntaxError
        except Exception:
            rdclass = dns.rdataclass.IN
        # Type
        rdtype = dns.rdatatype.from_text(token.value)
        (rdclass, rdtype, _, _) = self.message._parse_rr_header(
            section_number, name, rdclass, rdtype
        )
        self.message.find_rrset(
            section, name, rdclass, rdtype, create=True, force_unique=True
        )
        self.tok.get_eol()

    def _rr_line(self, section_number):
        """Process one line from the text format answer, authority, or
        additional data sections.
        """

        section = self.message.sections[section_number]
        # Name
        token = self.tok.get(want_leading=True)
        if not token.is_whitespace():
            self.last_name = self.tok.as_name(
                token, self.message.origin, self.relativize, self.relativize_to
            )
        name = self.last_name
        if name is None:
            raise NoPreviousName
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
        # TTL
        try:
            ttl = int(token.value, 0)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.exception.SyntaxError:
            raise dns.exception.SyntaxError
        except Exception:
            ttl = 0
        # Class
        try:
            rdclass = dns.rdataclass.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.exception.SyntaxError:
            raise dns.exception.SyntaxError
        except Exception:
            rdclass = dns.rdataclass.IN
        # Type
        rdtype = dns.rdatatype.from_text(token.value)
        (rdclass, rdtype, deleting, empty) = self.message._parse_rr_header(
            section_number, name, rdclass, rdtype
        )
        token = self.tok.get()
        if empty and not token.is_eol_or_eof():
            raise dns.exception.SyntaxError
        if not empty and token.is_eol_or_eof():
            raise dns.exception.UnexpectedEnd
        if not token.is_eol_or_eof():
            self.tok.unget(token)
            rd = dns.rdata.from_text(
                rdclass,
                rdtype,
                self.tok,
                self.message.origin,
                self.relativize,
                self.relativize_to,
            )
            covers = rd.covers()
        else:
            rd = None
            covers = dns.rdatatype.NONE
        rrset = self.message.find_rrset(
            section,
            name,
            rdclass,
            rdtype,
            covers,
            deleting,
            True,
            self.one_rr_per_rrset,
        )
        if rd is not None:
            rrset.add(rd, ttl)

    def _make_message(self):
        factory = _message_factory_from_opcode(self.opcode)
        message = factory(id=self.id)
        message.flags = self.flags
        if self.edns >= 0:
            message.use_edns(self.edns, self.ednsflags, self.payload)
        if self.rcode:
            message.set_rcode(self.rcode)
        if self.origin:
            message.origin = self.origin
        return message

    def read(self):
        """Read a text format DNS message and build a dns.message.Message
        object."""

        line_method = self._header_line
        section_number = None
        while 1:
            token = self.tok.get(True, True)
            if token.is_eol_or_eof():
                break
            if token.is_comment():
                u = token.value.upper()
                if u == "HEADER":
                    line_method = self._header_line

                if self.message:
                    message = self.message
                else:
                    # If we don't have a message, create one with the current
                    # opcode, so that we know which section names to parse.
                    message = self._make_message()
                try:
                    section_number = message._section_enum.from_text(u)
                    # We found a section name.  If we don't have a message,
                    # use the one we just created.
                    if not self.message:
                        self.message = message
                        self.one_rr_per_rrset = message._get_one_rr_per_rrset(
                            self.one_rr_per_rrset
                        )
                    if section_number == MessageSection.QUESTION:
                        line_method = self._question_line
                    else:
                        line_method = self._rr_line
                except Exception:
                    # It's just a comment.
                    pass
                self.tok.get_eol()
                continue
            self.tok.unget(token)
            line_method(section_number)
        if not self.message:
            self.message = self._make_message()
        return self.message


def from_text(
    text: str,
    idna_codec: Optional[dns.name.IDNACodec] = None,
    one_rr_per_rrset: bool = False,
    origin: Optional[dns.name.Name] = None,
    relativize: bool = True,
    relativize_to: Optional[dns.name.Name] = None,
) -> Message:
    """Convert the text format message into a message object.

    The reader stops after reading the first blank line in the input to
    facilitate reading multiple messages from a single file with
    ``dns.message.from_file()``.

    *text*, a ``str``, the text format message.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, then each RR is put
    into its own rrset.  The default is ``False``.

    *origin*, a ``dns.name.Name`` (or ``None``), the
    origin to use for relative names.

    *relativize*, a ``bool``.  If true, name will be relativized.

    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use
    when relativizing names.  If not set, the *origin* value will be used.

    Raises ``dns.message.UnknownHeaderField`` if a header is unknown.

    Raises ``dns.exception.SyntaxError`` if the text is badly formed.

    Returns a ``dns.message.Message object``
    """

    # 'text' can also be a file, but we don't publish that fact
    # since it's an implementation detail.  The official file
    # interface is from_file().

    reader = _TextReader(
        text, idna_codec, one_rr_per_rrset, origin, relativize, relativize_to
    )
    return reader.read()


def from_file(
    f: Any,
    idna_codec: Optional[dns.name.IDNACodec] = None,
    one_rr_per_rrset: bool = False,
) -> Message:
    """Read the next text format message from the specified file.

    Message blocks are separated by a single blank line.

    *f*, a ``file`` or ``str``.  If *f* is text, it is treated as the
    pathname of a file to open.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, then each RR is put
    into its own rrset.  The default is ``False``.

    Raises ``dns.message.UnknownHeaderField`` if a header is unknown.

    Raises ``dns.exception.SyntaxError`` if the text is badly formed.

    Returns a ``dns.message.Message object``
    """

    if isinstance(f, str):
        cm: contextlib.AbstractContextManager = open(f)
    else:
        cm = contextlib.nullcontext(f)
    with cm as f:
        return from_text(f, idna_codec, one_rr_per_rrset)
    assert False  # for mypy  lgtm[py/unreachable-statement]


def make_query(
    qname: Union[dns.name.Name, str],
    rdtype: Union[dns.rdatatype.RdataType, str],
    rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
    use_edns: Optional[Union[int, bool]] = None,
    want_dnssec: bool = False,
    ednsflags: Optional[int] = None,
    payload: Optional[int] = None,
    request_payload: Optional[int] = None,
    options: Optional[List[dns.edns.Option]] = None,
    idna_codec: Optional[dns.name.IDNACodec] = None,
    id: Optional[int] = None,
    flags: int = dns.flags.RD,
    pad: int = 0,
) -> QueryMessage:
    """Make a query message.

    The query name, type, and class may all be specified either
    as objects of the appropriate type, or as strings.

    The query will have a randomly chosen query id, and its DNS flags
    will be set to dns.flags.RD.

    qname, a ``dns.name.Name`` or ``str``, the query name.

    *rdtype*, an ``int`` or ``str``, the desired rdata type.

    *rdclass*, an ``int`` or ``str``,  the desired rdata class; the default
    is class IN.

    *use_edns*, an ``int``, ``bool`` or ``None``.  The EDNS level to use; the
    default is ``None``.  If ``None``, EDNS will be enabled only if other
    parameters (*ednsflags*, *payload*, *request_payload*, or *options*) are
    set.
    See the description of dns.message.Message.use_edns() for the possible
    values for use_edns and their meanings.

    *want_dnssec*, a ``bool``.  If ``True``, DNSSEC data is desired.

    *ednsflags*, an ``int``, the EDNS flag values.

    *payload*, an ``int``, is the EDNS sender's payload field, which is the
    maximum size of UDP datagram the sender can handle.  I.e. how big
    a response to this message can be.

    *request_payload*, an ``int``, is the EDNS payload size to use when
    sending this message.  If not specified, defaults to the value of
    *payload*.

    *options*, a list of ``dns.edns.Option`` objects or ``None``, the EDNS
    options.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.

    *id*, an ``int`` or ``None``, the desired query id.  The default is
    ``None``, which generates a random query id.

    *flags*, an ``int``, the desired query flags.  The default is
    ``dns.flags.RD``.

    *pad*, a non-negative ``int``.  If 0, the default, do not pad; otherwise add
    padding bytes to make the message size a multiple of *pad*.  Note that if
    padding is non-zero, an EDNS PADDING option will always be added to the
    message.

    Returns a ``dns.message.QueryMessage``
    """

    if isinstance(qname, str):
        qname = dns.name.from_text(qname, idna_codec=idna_codec)
    the_rdtype = dns.rdatatype.RdataType.make(rdtype)
    the_rdclass = dns.rdataclass.RdataClass.make(rdclass)
    m = QueryMessage(id=id)
    m.flags = dns.flags.Flag(flags)
    m.find_rrset(
        m.question, qname, the_rdclass, the_rdtype, create=True, force_unique=True
    )
    # only pass keywords on to use_edns if they have been set to a
    # non-None value.  Setting a field will turn EDNS on if it hasn't
    # been configured.
    kwargs: Dict[str, Any] = {}
    if ednsflags is not None:
        kwargs["ednsflags"] = ednsflags
    if payload is not None:
        kwargs["payload"] = payload
    if request_payload is not None:
        kwargs["request_payload"] = request_payload
    if options is not None:
        kwargs["options"] = options
    if kwargs and use_edns is None:
        use_edns = 0
    kwargs["edns"] = use_edns
    kwargs["pad"] = pad
    m.use_edns(**kwargs)
    m.want_dnssec(want_dnssec)
    return m


def make_response(
    query: Message,
    recursion_available: bool = False,
    our_payload: int = 8192,
    fudge: int = 300,
    tsig_error: int = 0,
) -> Message:
    """Make a message which is a response for the specified query.
    The message returned is really a response skeleton; it has all
    of the infrastructure required of a response, but none of the
    content.

    The response's question section is a shallow copy of the query's
    question section, so the query's question RRsets should not be
    changed.

    *query*, a ``dns.message.Message``, the query to respond to.

    *recursion_available*, a ``bool``, should RA be set in the response?

    *our_payload*, an ``int``, the payload size to advertise in EDNS
    responses.

    *fudge*, an ``int``, the TSIG time fudge.

    *tsig_error*, an ``int``, the TSIG error.

    Returns a ``dns.message.Message`` object whose specific class is
    appropriate for the query.  For example, if query is a
    ``dns.update.UpdateMessage``, response will be too.
    """

    if query.flags & dns.flags.QR:
        raise dns.exception.FormError("specified query message is not a query")
    factory = _message_factory_from_opcode(query.opcode())
    response = factory(id=query.id)
    response.flags = dns.flags.QR | (query.flags & dns.flags.RD)
    if recursion_available:
        response.flags |= dns.flags.RA
    response.set_opcode(query.opcode())
    response.question = list(query.question)
    if query.edns >= 0:
        response.use_edns(0, 0, our_payload, query.payload)
    if query.had_tsig:
        response.use_tsig(
            query.keyring,
            query.keyname,
            fudge,
            None,
            tsig_error,
            b"",
            query.keyalgorithm,
        )
        response.request_mac = query.mac
    return response


### BEGIN generated MessageSection constants

QUESTION = MessageSection.QUESTION
ANSWER = MessageSection.ANSWER
AUTHORITY = MessageSection.AUTHORITY
ADDITIONAL = MessageSection.ADDITIONAL

### END generated MessageSection constants
