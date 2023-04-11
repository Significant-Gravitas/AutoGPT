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

"""Asynchronous DNS stub resolver."""

from typing import Any, Dict, Optional, Union

import time

import dns.asyncbackend
import dns.asyncquery
import dns.exception
import dns.name
import dns.query
import dns.rdataclass
import dns.rdatatype
import dns.resolver  # lgtm[py/import-and-import-from]

# import some resolver symbols for brevity
from dns.resolver import NXDOMAIN, NoAnswer, NotAbsolute, NoRootSOA


# for indentation purposes below
_udp = dns.asyncquery.udp
_tcp = dns.asyncquery.tcp


class Resolver(dns.resolver.BaseResolver):
    """Asynchronous DNS stub resolver."""

    async def resolve(
        self,
        qname: Union[dns.name.Name, str],
        rdtype: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.A,
        rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
        tcp: bool = False,
        source: Optional[str] = None,
        raise_on_no_answer: bool = True,
        source_port: int = 0,
        lifetime: Optional[float] = None,
        search: Optional[bool] = None,
        backend: Optional[dns.asyncbackend.Backend] = None,
    ) -> dns.resolver.Answer:
        """Query nameservers asynchronously to find the answer to the question.

        *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
        the default, then dnspython will use the default backend.

        See :py:func:`dns.resolver.Resolver.resolve()` for the
        documentation of the other parameters, exceptions, and return
        type of this method.
        """

        resolution = dns.resolver._Resolution(
            self, qname, rdtype, rdclass, tcp, raise_on_no_answer, search
        )
        if not backend:
            backend = dns.asyncbackend.get_default_backend()
        start = time.time()
        while True:
            (request, answer) = resolution.next_request()
            # Note we need to say "if answer is not None" and not just
            # "if answer" because answer implements __len__, and python
            # will call that.  We want to return if we have an answer
            # object, including in cases where its length is 0.
            if answer is not None:
                # cache hit!
                return answer
            assert request is not None  # needed for type checking
            done = False
            while not done:
                (nameserver, port, tcp, backoff) = resolution.next_nameserver()
                if backoff:
                    await backend.sleep(backoff)
                timeout = self._compute_timeout(start, lifetime, resolution.errors)
                try:
                    if dns.inet.is_address(nameserver):
                        if tcp:
                            response = await _tcp(
                                request,
                                nameserver,
                                timeout,
                                port,
                                source,
                                source_port,
                                backend=backend,
                            )
                        else:
                            response = await _udp(
                                request,
                                nameserver,
                                timeout,
                                port,
                                source,
                                source_port,
                                raise_on_truncation=True,
                                backend=backend,
                            )
                    else:
                        response = await dns.asyncquery.https(
                            request, nameserver, timeout=timeout
                        )
                except Exception as ex:
                    (_, done) = resolution.query_result(None, ex)
                    continue
                (answer, done) = resolution.query_result(response, None)
                # Note we need to say "if answer is not None" and not just
                # "if answer" because answer implements __len__, and python
                # will call that.  We want to return if we have an answer
                # object, including in cases where its length is 0.
                if answer is not None:
                    return answer

    async def resolve_address(
        self, ipaddr: str, *args: Any, **kwargs: Any
    ) -> dns.resolver.Answer:
        """Use an asynchronous resolver to run a reverse query for PTR
        records.

        This utilizes the resolve() method to perform a PTR lookup on the
        specified IP address.

        *ipaddr*, a ``str``, the IPv4 or IPv6 address you want to get
        the PTR record for.

        All other arguments that can be passed to the resolve() function
        except for rdtype and rdclass are also supported by this
        function.

        """
        # We make a modified kwargs for type checking happiness, as otherwise
        # we get a legit warning about possibly having rdtype and rdclass
        # in the kwargs more than once.
        modified_kwargs: Dict[str, Any] = {}
        modified_kwargs.update(kwargs)
        modified_kwargs["rdtype"] = dns.rdatatype.PTR
        modified_kwargs["rdclass"] = dns.rdataclass.IN
        return await self.resolve(
            dns.reversename.from_address(ipaddr), *args, **modified_kwargs
        )

    # pylint: disable=redefined-outer-name

    async def canonical_name(self, name: Union[dns.name.Name, str]) -> dns.name.Name:
        """Determine the canonical name of *name*.

        The canonical name is the name the resolver uses for queries
        after all CNAME and DNAME renamings have been applied.

        *name*, a ``dns.name.Name`` or ``str``, the query name.

        This method can raise any exception that ``resolve()`` can
        raise, other than ``dns.resolver.NoAnswer`` and
        ``dns.resolver.NXDOMAIN``.

        Returns a ``dns.name.Name``.
        """
        try:
            answer = await self.resolve(name, raise_on_no_answer=False)
            canonical_name = answer.canonical_name
        except dns.resolver.NXDOMAIN as e:
            canonical_name = e.canonical_name
        return canonical_name


default_resolver = None


def get_default_resolver() -> Resolver:
    """Get the default asynchronous resolver, initializing it if necessary."""
    if default_resolver is None:
        reset_default_resolver()
    assert default_resolver is not None
    return default_resolver


def reset_default_resolver() -> None:
    """Re-initialize default asynchronous resolver.

    Note that the resolver configuration (i.e. /etc/resolv.conf on UNIX
    systems) will be re-read immediately.
    """

    global default_resolver
    default_resolver = Resolver()


async def resolve(
    qname: Union[dns.name.Name, str],
    rdtype: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.A,
    rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
    tcp: bool = False,
    source: Optional[str] = None,
    raise_on_no_answer: bool = True,
    source_port: int = 0,
    lifetime: Optional[float] = None,
    search: Optional[bool] = None,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> dns.resolver.Answer:
    """Query nameservers asynchronously to find the answer to the question.

    This is a convenience function that uses the default resolver
    object to make the query.

    See :py:func:`dns.asyncresolver.Resolver.resolve` for more
    information on the parameters.
    """

    return await get_default_resolver().resolve(
        qname,
        rdtype,
        rdclass,
        tcp,
        source,
        raise_on_no_answer,
        source_port,
        lifetime,
        search,
        backend,
    )


async def resolve_address(
    ipaddr: str, *args: Any, **kwargs: Any
) -> dns.resolver.Answer:
    """Use a resolver to run a reverse query for PTR records.

    See :py:func:`dns.asyncresolver.Resolver.resolve_address` for more
    information on the parameters.
    """

    return await get_default_resolver().resolve_address(ipaddr, *args, **kwargs)


async def canonical_name(name: Union[dns.name.Name, str]) -> dns.name.Name:
    """Determine the canonical name of *name*.

    See :py:func:`dns.resolver.Resolver.canonical_name` for more
    information on the parameters and possible exceptions.
    """

    return await get_default_resolver().canonical_name(name)


async def zone_for_name(
    name: Union[dns.name.Name, str],
    rdclass: dns.rdataclass.RdataClass = dns.rdataclass.IN,
    tcp: bool = False,
    resolver: Optional[Resolver] = None,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> dns.name.Name:
    """Find the name of the zone which contains the specified name.

    See :py:func:`dns.resolver.Resolver.zone_for_name` for more
    information on the parameters and possible exceptions.
    """

    if isinstance(name, str):
        name = dns.name.from_text(name, dns.name.root)
    if resolver is None:
        resolver = get_default_resolver()
    if not name.is_absolute():
        raise NotAbsolute(name)
    while True:
        try:
            answer = await resolver.resolve(
                name, dns.rdatatype.SOA, rdclass, tcp, backend=backend
            )
            assert answer.rrset is not None
            if answer.rrset.name == name:
                return name
            # otherwise we were CNAMEd or DNAMEd and need to look higher
        except (NXDOMAIN, NoAnswer):
            pass
        try:
            name = name.parent()
        except dns.name.NoParent:  # pragma: no cover
            raise NoRootSOA
