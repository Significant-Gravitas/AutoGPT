# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2006-2017 Nominum, Inc.
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

"""DNS E.164 helpers."""

from typing import Iterable, Optional, Union

import dns.exception
import dns.name
import dns.resolver

#: The public E.164 domain.
public_enum_domain = dns.name.from_text("e164.arpa.")


def from_e164(
    text: str, origin: Optional[dns.name.Name] = public_enum_domain
) -> dns.name.Name:
    """Convert an E.164 number in textual form into a Name object whose
    value is the ENUM domain name for that number.

    Non-digits in the text are ignored, i.e. "16505551212",
    "+1.650.555.1212" and "1 (650) 555-1212" are all the same.

    *text*, a ``str``, is an E.164 number in textual form.

    *origin*, a ``dns.name.Name``, the domain in which the number
    should be constructed.  The default is ``e164.arpa.``.

    Returns a ``dns.name.Name``.
    """

    parts = [d for d in text if d.isdigit()]
    parts.reverse()
    return dns.name.from_text(".".join(parts), origin=origin)


def to_e164(
    name: dns.name.Name,
    origin: Optional[dns.name.Name] = public_enum_domain,
    want_plus_prefix: bool = True,
) -> str:
    """Convert an ENUM domain name into an E.164 number.

    Note that dnspython does not have any information about preferred
    number formats within national numbering plans, so all numbers are
    emitted as a simple string of digits, prefixed by a '+' (unless
    *want_plus_prefix* is ``False``).

    *name* is a ``dns.name.Name``, the ENUM domain name.

    *origin* is a ``dns.name.Name``, a domain containing the ENUM
    domain name.  The name is relativized to this domain before being
    converted to text.  If ``None``, no relativization is done.

    *want_plus_prefix* is a ``bool``.  If True, add a '+' to the beginning of
    the returned number.

    Returns a ``str``.

    """
    if origin is not None:
        name = name.relativize(origin)
    dlabels = [d for d in name.labels if d.isdigit() and len(d) == 1]
    if len(dlabels) != len(name.labels):
        raise dns.exception.SyntaxError("non-digit labels in ENUM domain name")
    dlabels.reverse()
    text = b"".join(dlabels)
    if want_plus_prefix:
        text = b"+" + text
    return text.decode()


def query(
    number: str,
    domains: Iterable[Union[dns.name.Name, str]],
    resolver: Optional[dns.resolver.Resolver] = None,
) -> dns.resolver.Answer:
    """Look for NAPTR RRs for the specified number in the specified domains.

    e.g. lookup('16505551212', ['e164.dnspython.org.', 'e164.arpa.'])

    *number*, a ``str`` is the number to look for.

    *domains* is an iterable containing ``dns.name.Name`` values.

    *resolver*, a ``dns.resolver.Resolver``, is the resolver to use.  If
    ``None``, the default resolver is used.
    """

    if resolver is None:
        resolver = dns.resolver.get_default_resolver()
    e_nx = dns.resolver.NXDOMAIN()
    for domain in domains:
        if isinstance(domain, str):
            domain = dns.name.from_text(domain)
        qname = dns.e164.from_e164(number, domain)
        try:
            return resolver.resolve(qname, "NAPTR")
        except dns.resolver.NXDOMAIN as e:
            e_nx += e
    raise e_nx
