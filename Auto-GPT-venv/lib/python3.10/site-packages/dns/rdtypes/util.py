# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2006, 2007, 2009-2011 Nominum, Inc.
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

import collections
import random
import struct

import dns.exception
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata


class Gateway:
    """A helper class for the IPSECKEY gateway and AMTRELAY relay fields"""

    name = ""

    def __init__(self, type, gateway=None):
        self.type = dns.rdata.Rdata._as_uint8(type)
        self.gateway = gateway
        self._check()

    @classmethod
    def _invalid_type(cls, gateway_type):
        return f"invalid {cls.name} type: {gateway_type}"

    def _check(self):
        if self.type == 0:
            if self.gateway not in (".", None):
                raise SyntaxError(f"invalid {self.name} for type 0")
            self.gateway = None
        elif self.type == 1:
            # check that it's OK
            dns.ipv4.inet_aton(self.gateway)
        elif self.type == 2:
            # check that it's OK
            dns.ipv6.inet_aton(self.gateway)
        elif self.type == 3:
            if not isinstance(self.gateway, dns.name.Name):
                raise SyntaxError(f"invalid {self.name}; not a name")
        else:
            raise SyntaxError(self._invalid_type(self.type))

    def to_text(self, origin=None, relativize=True):
        if self.type == 0:
            return "."
        elif self.type in (1, 2):
            return self.gateway
        elif self.type == 3:
            return str(self.gateway.choose_relativity(origin, relativize))
        else:
            raise ValueError(self._invalid_type(self.type))  # pragma: no cover

    @classmethod
    def from_text(
        cls, gateway_type, tok, origin=None, relativize=True, relativize_to=None
    ):
        if gateway_type in (0, 1, 2):
            gateway = tok.get_string()
        elif gateway_type == 3:
            gateway = tok.get_name(origin, relativize, relativize_to)
        else:
            raise dns.exception.SyntaxError(
                cls._invalid_type(gateway_type)
            )  # pragma: no cover
        return cls(gateway_type, gateway)

    # pylint: disable=unused-argument
    def to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if self.type == 0:
            pass
        elif self.type == 1:
            file.write(dns.ipv4.inet_aton(self.gateway))
        elif self.type == 2:
            file.write(dns.ipv6.inet_aton(self.gateway))
        elif self.type == 3:
            self.gateway.to_wire(file, None, origin, False)
        else:
            raise ValueError(self._invalid_type(self.type))  # pragma: no cover

    # pylint: enable=unused-argument

    @classmethod
    def from_wire_parser(cls, gateway_type, parser, origin=None):
        if gateway_type == 0:
            gateway = None
        elif gateway_type == 1:
            gateway = dns.ipv4.inet_ntoa(parser.get_bytes(4))
        elif gateway_type == 2:
            gateway = dns.ipv6.inet_ntoa(parser.get_bytes(16))
        elif gateway_type == 3:
            gateway = parser.get_name(origin)
        else:
            raise dns.exception.FormError(cls._invalid_type(gateway_type))
        return cls(gateway_type, gateway)


class Bitmap:
    """A helper class for the NSEC/NSEC3/CSYNC type bitmaps"""

    type_name = ""

    def __init__(self, windows=None):
        last_window = -1
        self.windows = windows
        for (window, bitmap) in self.windows:
            if not isinstance(window, int):
                raise ValueError(f"bad {self.type_name} window type")
            if window <= last_window:
                raise ValueError(f"bad {self.type_name} window order")
            if window > 256:
                raise ValueError(f"bad {self.type_name} window number")
            last_window = window
            if not isinstance(bitmap, bytes):
                raise ValueError(f"bad {self.type_name} octets type")
            if len(bitmap) == 0 or len(bitmap) > 32:
                raise ValueError(f"bad {self.type_name} octets")

    def to_text(self):
        text = ""
        for (window, bitmap) in self.windows:
            bits = []
            for (i, byte) in enumerate(bitmap):
                for j in range(0, 8):
                    if byte & (0x80 >> j):
                        rdtype = window * 256 + i * 8 + j
                        bits.append(dns.rdatatype.to_text(rdtype))
            text += " " + " ".join(bits)
        return text

    @classmethod
    def from_text(cls, tok):
        rdtypes = []
        for token in tok.get_remaining():
            rdtype = dns.rdatatype.from_text(token.unescape().value)
            if rdtype == 0:
                raise dns.exception.SyntaxError(f"{cls.type_name} with bit 0")
            rdtypes.append(rdtype)
        rdtypes.sort()
        window = 0
        octets = 0
        prior_rdtype = 0
        bitmap = bytearray(b"\0" * 32)
        windows = []
        for rdtype in rdtypes:
            if rdtype == prior_rdtype:
                continue
            prior_rdtype = rdtype
            new_window = rdtype // 256
            if new_window != window:
                if octets != 0:
                    windows.append((window, bytes(bitmap[0:octets])))
                bitmap = bytearray(b"\0" * 32)
                window = new_window
            offset = rdtype % 256
            byte = offset // 8
            bit = offset % 8
            octets = byte + 1
            bitmap[byte] = bitmap[byte] | (0x80 >> bit)
        if octets != 0:
            windows.append((window, bytes(bitmap[0:octets])))
        return cls(windows)

    def to_wire(self, file):
        for (window, bitmap) in self.windows:
            file.write(struct.pack("!BB", window, len(bitmap)))
            file.write(bitmap)

    @classmethod
    def from_wire_parser(cls, parser):
        windows = []
        while parser.remaining() > 0:
            window = parser.get_uint8()
            bitmap = parser.get_counted_bytes()
            windows.append((window, bitmap))
        return cls(windows)


def _priority_table(items):
    by_priority = collections.defaultdict(list)
    for rdata in items:
        by_priority[rdata._processing_priority()].append(rdata)
    return by_priority


def priority_processing_order(iterable):
    items = list(iterable)
    if len(items) == 1:
        return items
    by_priority = _priority_table(items)
    ordered = []
    for k in sorted(by_priority.keys()):
        rdatas = by_priority[k]
        random.shuffle(rdatas)
        ordered.extend(rdatas)
    return ordered


_no_weight = 0.1


def weighted_processing_order(iterable):
    items = list(iterable)
    if len(items) == 1:
        return items
    by_priority = _priority_table(items)
    ordered = []
    for k in sorted(by_priority.keys()):
        rdatas = by_priority[k]
        total = sum(rdata._processing_weight() or _no_weight for rdata in rdatas)
        while len(rdatas) > 1:
            r = random.uniform(0, total)
            for (n, rdata) in enumerate(rdatas):
                weight = rdata._processing_weight() or _no_weight
                if weight > r:
                    break
                r -= weight
            total -= weight
            ordered.append(rdata)  # pylint: disable=undefined-loop-variable
            del rdatas[n]  # pylint: disable=undefined-loop-variable
        ordered.append(rdatas[0])
    return ordered


def parse_formatted_hex(formatted, num_chunks, chunk_size, separator):
    if len(formatted) != num_chunks * (chunk_size + 1) - 1:
        raise ValueError("invalid formatted hex string")
    value = b""
    for _ in range(num_chunks):
        chunk = formatted[0:chunk_size]
        value += int(chunk, 16).to_bytes(chunk_size // 2, "big")
        formatted = formatted[chunk_size:]
        if len(formatted) > 0 and formatted[0] != separator:
            raise ValueError("invalid formatted hex string")
        formatted = formatted[1:]
    return value
