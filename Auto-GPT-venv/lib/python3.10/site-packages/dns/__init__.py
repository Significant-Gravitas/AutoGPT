# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2007, 2009, 2011 Nominum, Inc.
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

"""dnspython DNS toolkit"""

__all__ = [
    "asyncbackend",
    "asyncquery",
    "asyncresolver",
    "dnssec",
    "dnssectypes",
    "e164",
    "edns",
    "entropy",
    "exception",
    "flags",
    "immutable",
    "inet",
    "ipv4",
    "ipv6",
    "message",
    "name",
    "namedict",
    "node",
    "opcode",
    "query",
    "quic",
    "rcode",
    "rdata",
    "rdataclass",
    "rdataset",
    "rdatatype",
    "renderer",
    "resolver",
    "reversename",
    "rrset",
    "serial",
    "set",
    "tokenizer",
    "transaction",
    "tsig",
    "tsigkeyring",
    "ttl",
    "rdtypes",
    "update",
    "version",
    "versioned",
    "wire",
    "xfr",
    "zone",
    "zonetypes",
    "zonefile",
]

from dns.version import version as __version__  # noqa
