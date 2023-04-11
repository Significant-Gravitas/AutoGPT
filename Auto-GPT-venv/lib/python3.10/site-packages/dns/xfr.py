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

from typing import Any, List, Optional, Tuple, Union

import dns.exception
import dns.message
import dns.name
import dns.rcode
import dns.serial
import dns.rdataset
import dns.rdatatype
import dns.transaction
import dns.tsig
import dns.zone


class TransferError(dns.exception.DNSException):
    """A zone transfer response got a non-zero rcode."""

    def __init__(self, rcode):
        message = "Zone transfer error: %s" % dns.rcode.to_text(rcode)
        super().__init__(message)
        self.rcode = rcode


class SerialWentBackwards(dns.exception.FormError):
    """The current serial number is less than the serial we know."""


class UseTCP(dns.exception.DNSException):
    """This IXFR cannot be completed with UDP."""


class Inbound:
    """
    State machine for zone transfers.
    """

    def __init__(
        self,
        txn_manager: dns.transaction.TransactionManager,
        rdtype: dns.rdatatype.RdataType = dns.rdatatype.AXFR,
        serial: Optional[int] = None,
        is_udp: bool = False,
    ):
        """Initialize an inbound zone transfer.

        *txn_manager* is a :py:class:`dns.transaction.TransactionManager`.

        *rdtype* can be `dns.rdatatype.AXFR` or `dns.rdatatype.IXFR`

        *serial* is the base serial number for IXFRs, and is required in
        that case.

        *is_udp*, a ``bool`` indidicates if UDP is being used for this
        XFR.
        """
        self.txn_manager = txn_manager
        self.txn: Optional[dns.transaction.Transaction] = None
        self.rdtype = rdtype
        if rdtype == dns.rdatatype.IXFR:
            if serial is None:
                raise ValueError("a starting serial must be supplied for IXFRs")
        elif is_udp:
            raise ValueError("is_udp specified for AXFR")
        self.serial = serial
        self.is_udp = is_udp
        (_, _, self.origin) = txn_manager.origin_information()
        self.soa_rdataset: Optional[dns.rdataset.Rdataset] = None
        self.done = False
        self.expecting_SOA = False
        self.delete_mode = False

    def process_message(self, message: dns.message.Message) -> bool:
        """Process one message in the transfer.

        The message should have the same relativization as was specified when
        the `dns.xfr.Inbound` was created.  The message should also have been
        created with `one_rr_per_rrset=True` because order matters.

        Returns `True` if the transfer is complete, and `False` otherwise.
        """
        if self.txn is None:
            replacement = self.rdtype == dns.rdatatype.AXFR
            self.txn = self.txn_manager.writer(replacement)
        rcode = message.rcode()
        if rcode != dns.rcode.NOERROR:
            raise TransferError(rcode)
        #
        # We don't require a question section, but if it is present is
        # should be correct.
        #
        if len(message.question) > 0:
            if message.question[0].name != self.origin:
                raise dns.exception.FormError("wrong question name")
            if message.question[0].rdtype != self.rdtype:
                raise dns.exception.FormError("wrong question rdatatype")
        answer_index = 0
        if self.soa_rdataset is None:
            #
            # This is the first message.  We're expecting an SOA at
            # the origin.
            #
            if not message.answer or message.answer[0].name != self.origin:
                raise dns.exception.FormError("No answer or RRset not for zone origin")
            rrset = message.answer[0]
            rdataset = rrset
            if rdataset.rdtype != dns.rdatatype.SOA:
                raise dns.exception.FormError("first RRset is not an SOA")
            answer_index = 1
            self.soa_rdataset = rdataset.copy()
            if self.rdtype == dns.rdatatype.IXFR:
                if self.soa_rdataset[0].serial == self.serial:
                    #
                    # We're already up-to-date.
                    #
                    self.done = True
                elif dns.serial.Serial(self.soa_rdataset[0].serial) < self.serial:
                    # It went backwards!
                    raise SerialWentBackwards
                else:
                    if self.is_udp and len(message.answer[answer_index:]) == 0:
                        #
                        # There are no more records, so this is the
                        # "truncated" response.  Say to use TCP
                        #
                        raise UseTCP
                    #
                    # Note we're expecting another SOA so we can detect
                    # if this IXFR response is an AXFR-style response.
                    #
                    self.expecting_SOA = True
        #
        # Process the answer section (other than the initial SOA in
        # the first message).
        #
        for rrset in message.answer[answer_index:]:
            name = rrset.name
            rdataset = rrset
            if self.done:
                raise dns.exception.FormError("answers after final SOA")
            assert self.txn is not None  # for mypy
            if rdataset.rdtype == dns.rdatatype.SOA and name == self.origin:
                #
                # Every time we see an origin SOA delete_mode inverts
                #
                if self.rdtype == dns.rdatatype.IXFR:
                    self.delete_mode = not self.delete_mode
                #
                # If this SOA Rdataset is equal to the first we saw
                # then we're finished. If this is an IXFR we also
                # check that we're seeing the record in the expected
                # part of the response.
                #
                if rdataset == self.soa_rdataset and (
                    self.rdtype == dns.rdatatype.AXFR
                    or (self.rdtype == dns.rdatatype.IXFR and self.delete_mode)
                ):
                    #
                    # This is the final SOA
                    #
                    if self.expecting_SOA:
                        # We got an empty IXFR sequence!
                        raise dns.exception.FormError("empty IXFR sequence")
                    if (
                        self.rdtype == dns.rdatatype.IXFR
                        and self.serial != rdataset[0].serial
                    ):
                        raise dns.exception.FormError("unexpected end of IXFR sequence")
                    self.txn.replace(name, rdataset)
                    self.txn.commit()
                    self.txn = None
                    self.done = True
                else:
                    #
                    # This is not the final SOA
                    #
                    self.expecting_SOA = False
                    if self.rdtype == dns.rdatatype.IXFR:
                        if self.delete_mode:
                            # This is the start of an IXFR deletion set
                            if rdataset[0].serial != self.serial:
                                raise dns.exception.FormError(
                                    "IXFR base serial mismatch"
                                )
                        else:
                            # This is the start of an IXFR addition set
                            self.serial = rdataset[0].serial
                            self.txn.replace(name, rdataset)
                    else:
                        # We saw a non-final SOA for the origin in an AXFR.
                        raise dns.exception.FormError("unexpected origin SOA in AXFR")
                continue
            if self.expecting_SOA:
                #
                # We made an IXFR request and are expecting another
                # SOA RR, but saw something else, so this must be an
                # AXFR response.
                #
                self.rdtype = dns.rdatatype.AXFR
                self.expecting_SOA = False
                self.delete_mode = False
                self.txn.rollback()
                self.txn = self.txn_manager.writer(True)
                #
                # Note we are falling through into the code below
                # so whatever rdataset this was gets written.
                #
            # Add or remove the data
            if self.delete_mode:
                self.txn.delete_exact(name, rdataset)
            else:
                self.txn.add(name, rdataset)
        if self.is_udp and not self.done:
            #
            # This is a UDP IXFR and we didn't get to done, and we didn't
            # get the proper "truncated" response
            #
            raise dns.exception.FormError("unexpected end of UDP IXFR")
        return self.done

    #
    # Inbounds are context managers.
    #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.txn:
            self.txn.rollback()
        return False


def make_query(
    txn_manager: dns.transaction.TransactionManager,
    serial: Optional[int] = 0,
    use_edns: Optional[Union[int, bool]] = None,
    ednsflags: Optional[int] = None,
    payload: Optional[int] = None,
    request_payload: Optional[int] = None,
    options: Optional[List[dns.edns.Option]] = None,
    keyring: Any = None,
    keyname: Optional[dns.name.Name] = None,
    keyalgorithm: Union[dns.name.Name, str] = dns.tsig.default_algorithm,
) -> Tuple[dns.message.QueryMessage, Optional[int]]:
    """Make an AXFR or IXFR query.

    *txn_manager* is a ``dns.transaction.TransactionManager``, typically a
    ``dns.zone.Zone``.

    *serial* is an ``int`` or ``None``.  If 0, then IXFR will be
    attempted using the most recent serial number from the
    *txn_manager*; it is the caller's responsibility to ensure there
    are no write transactions active that could invalidate the
    retrieved serial.  If a serial cannot be determined, AXFR will be
    forced.  Other integer values are the starting serial to use.
    ``None`` forces an AXFR.

    Please see the documentation for :py:func:`dns.message.make_query` and
    :py:func:`dns.message.Message.use_tsig` for details on the other parameters
    to this function.

    Returns a `(query, serial)` tuple.
    """
    (zone_origin, _, origin) = txn_manager.origin_information()
    if zone_origin is None:
        raise ValueError("no zone origin")
    if serial is None:
        rdtype = dns.rdatatype.AXFR
    elif not isinstance(serial, int):
        raise ValueError("serial is not an integer")
    elif serial == 0:
        with txn_manager.reader() as txn:
            rdataset = txn.get(origin, "SOA")
            if rdataset:
                serial = rdataset[0].serial
                rdtype = dns.rdatatype.IXFR
            else:
                serial = None
                rdtype = dns.rdatatype.AXFR
    elif serial > 0 and serial < 4294967296:
        rdtype = dns.rdatatype.IXFR
    else:
        raise ValueError("serial out-of-range")
    rdclass = txn_manager.get_class()
    q = dns.message.make_query(
        zone_origin,
        rdtype,
        rdclass,
        use_edns,
        False,
        ednsflags,
        payload,
        request_payload,
        options,
    )
    if serial is not None:
        rdata = dns.rdata.from_text(rdclass, "SOA", f". . {serial} 0 0 0 0")
        rrset = q.find_rrset(
            q.authority, zone_origin, rdclass, dns.rdatatype.SOA, create=True
        )
        rrset.add(rdata, 0)
    if keyring is not None:
        q.use_tsig(keyring, keyname, algorithm=keyalgorithm)
    return (q, serial)


def extract_serial_from_query(query: dns.message.Message) -> Optional[int]:
    """Extract the SOA serial number from query if it is an IXFR and return
    it, otherwise return None.

    *query* is a dns.message.QueryMessage that is an IXFR or AXFR request.

    Raises if the query is not an IXFR or AXFR, or if an IXFR doesn't have
    an appropriate SOA RRset in the authority section.
    """
    if not isinstance(query, dns.message.QueryMessage):
        raise ValueError("query not a QueryMessage")
    question = query.question[0]
    if question.rdtype == dns.rdatatype.AXFR:
        return None
    elif question.rdtype != dns.rdatatype.IXFR:
        raise ValueError("query is not an AXFR or IXFR")
    soa = query.find_rrset(
        query.authority, question.name, question.rdclass, dns.rdatatype.SOA
    )
    return soa[0].serial
