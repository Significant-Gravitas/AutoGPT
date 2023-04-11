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

"""DNS stub resolver."""

from typing import Any, Dict, List, Optional, Tuple, Union

from urllib.parse import urlparse
import contextlib
import socket
import sys
import threading
import time
import random
import warnings

import dns.exception
import dns.edns
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig

if sys.platform == "win32":
    import dns.win32util


class NXDOMAIN(dns.exception.DNSException):
    """The DNS query name does not exist."""

    supp_kwargs = {"qnames", "responses"}
    fmt = None  # we have our own __str__ implementation

    # pylint: disable=arguments-differ

    # We do this as otherwise mypy complains about unexpected keyword argument
    # idna_exception
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_kwargs(self, qnames, responses=None):
        if not isinstance(qnames, (list, tuple, set)):
            raise AttributeError("qnames must be a list, tuple or set")
        if len(qnames) == 0:
            raise AttributeError("qnames must contain at least one element")
        if responses is None:
            responses = {}
        elif not isinstance(responses, dict):
            raise AttributeError("responses must be a dict(qname=response)")
        kwargs = dict(qnames=qnames, responses=responses)
        return kwargs

    def __str__(self):
        if "qnames" not in self.kwargs:
            return super().__str__()
        qnames = self.kwargs["qnames"]
        if len(qnames) > 1:
            msg = "None of DNS query names exist"
        else:
            msg = "The DNS query name does not exist"
        qnames = ", ".join(map(str, qnames))
        return "{}: {}".format(msg, qnames)

    @property
    def canonical_name(self):
        """Return the unresolved canonical name."""
        if "qnames" not in self.kwargs:
            raise TypeError("parametrized exception required")
        for qname in self.kwargs["qnames"]:
            response = self.kwargs["responses"][qname]
            try:
                cname = response.canonical_name()
                if cname != qname:
                    return cname
            except Exception:
                # We can just eat this exception as it means there was
                # something wrong with the response.
                pass
        return self.kwargs["qnames"][0]

    def __add__(self, e_nx):
        """Augment by results from another NXDOMAIN exception."""
        qnames0 = list(self.kwargs.get("qnames", []))
        responses0 = dict(self.kwargs.get("responses", {}))
        responses1 = e_nx.kwargs.get("responses", {})
        for qname1 in e_nx.kwargs.get("qnames", []):
            if qname1 not in qnames0:
                qnames0.append(qname1)
            if qname1 in responses1:
                responses0[qname1] = responses1[qname1]
        return NXDOMAIN(qnames=qnames0, responses=responses0)

    def qnames(self):
        """All of the names that were tried.

        Returns a list of ``dns.name.Name``.
        """
        return self.kwargs["qnames"]

    def responses(self):
        """A map from queried names to their NXDOMAIN responses.

        Returns a dict mapping a ``dns.name.Name`` to a
        ``dns.message.Message``.
        """
        return self.kwargs["responses"]

    def response(self, qname):
        """The response for query *qname*.

        Returns a ``dns.message.Message``.
        """
        return self.kwargs["responses"][qname]


class YXDOMAIN(dns.exception.DNSException):
    """The DNS query name is too long after DNAME substitution."""


ErrorTuple = Tuple[
    Optional[str], bool, int, Union[Exception, str], Optional[dns.message.Message]
]


def _errors_to_text(errors: List[ErrorTuple]) -> List[str]:
    """Turn a resolution errors trace into a list of text."""
    texts = []
    for err in errors:
        texts.append(
            "Server {} {} port {} answered {}".format(
                err[0], "TCP" if err[1] else "UDP", err[2], err[3]
            )
        )
    return texts


class LifetimeTimeout(dns.exception.Timeout):
    """The resolution lifetime expired."""

    msg = "The resolution lifetime expired."
    fmt = "%s after {timeout:.3f} seconds: {errors}" % msg[:-1]
    supp_kwargs = {"timeout", "errors"}

    # We do this as otherwise mypy complains about unexpected keyword argument
    # idna_exception
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fmt_kwargs(self, **kwargs):
        srv_msgs = _errors_to_text(kwargs["errors"])
        return super()._fmt_kwargs(
            timeout=kwargs["timeout"], errors="; ".join(srv_msgs)
        )


# We added more detail to resolution timeouts, but they are still
# subclasses of dns.exception.Timeout for backwards compatibility.  We also
# keep dns.resolver.Timeout defined for backwards compatibility.
Timeout = LifetimeTimeout


class NoAnswer(dns.exception.DNSException):
    """The DNS response does not contain an answer to the question."""

    fmt = "The DNS response does not contain an answer " + "to the question: {query}"
    supp_kwargs = {"response"}

    # We do this as otherwise mypy complains about unexpected keyword argument
    # idna_exception
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fmt_kwargs(self, **kwargs):
        return super()._fmt_kwargs(query=kwargs["response"].question)

    def response(self):
        return self.kwargs["response"]


class NoNameservers(dns.exception.DNSException):
    """All nameservers failed to answer the query.

    errors: list of servers and respective errors
    The type of errors is
    [(server IP address, any object convertible to string)].
    Non-empty errors list will add explanatory message ()
    """

    msg = "All nameservers failed to answer the query."
    fmt = "%s {query}: {errors}" % msg[:-1]
    supp_kwargs = {"request", "errors"}

    # We do this as otherwise mypy complains about unexpected keyword argument
    # idna_exception
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fmt_kwargs(self, **kwargs):
        srv_msgs = _errors_to_text(kwargs["errors"])
        return super()._fmt_kwargs(
            query=kwargs["request"].question, errors="; ".join(srv_msgs)
        )


class NotAbsolute(dns.exception.DNSException):
    """An absolute domain name is required but a relative name was provided."""


class NoRootSOA(dns.exception.DNSException):
    """There is no SOA RR at the DNS root name. This should never happen!"""


class NoMetaqueries(dns.exception.DNSException):
    """DNS metaqueries are not allowed."""


class NoResolverConfiguration(dns.exception.DNSException):
    """Resolver configuration could not be read or specified no nameservers."""


class Answer:
    """DNS stub resolver answer.

    Instances of this class bundle up the result of a successful DNS
    resolution.

    For convenience, the answer object implements much of the sequence
    protocol, forwarding to its ``rrset`` attribute.  E.g.
    ``for a in answer`` is equivalent to ``for a in answer.rrset``.
    ``answer[i]`` is equivalent to ``answer.rrset[i]``, and
    ``answer[i:j]`` is equivalent to ``answer.rrset[i:j]``.

    Note that CNAMEs or DNAMEs in the response may mean that answer
    RRset's name might not be the query name.
    """

    def __init__(
        self,
        qname: dns.name.Name,
        rdtype: dns.rdatatype.RdataType,
        rdclass: dns.rdataclass.RdataClass,
        response: dns.message.QueryMessage,
        nameserver: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.qname = qname
        self.rdtype = rdtype
        self.rdclass = rdclass
        self.response = response
        self.nameserver = nameserver
        self.port = port
        self.chaining_result = response.resolve_chaining()
        # Copy some attributes out of chaining_result for backwards
        # compatibility and convenience.
        self.canonical_name = self.chaining_result.canonical_name
        self.rrset = self.chaining_result.answer
        self.expiration = time.time() + self.chaining_result.minimum_ttl

    def __getattr__(self, attr):  # pragma: no cover
        if attr == "name":
            return self.rrset.name
        elif attr == "ttl":
            return self.rrset.ttl
        elif attr == "covers":
            return self.rrset.covers
        elif attr == "rdclass":
            return self.rrset.rdclass
        elif attr == "rdtype":
            return self.rrset.rdtype
        else:
            raise AttributeError(attr)

    def __len__(self):
        return self.rrset and len(self.rrset) or 0

    def __iter__(self):
        return self.rrset and iter(self.rrset) or iter(tuple())

    def __getitem__(self, i):
        if self.rrset is None:
            raise IndexError
        return self.rrset[i]

    def __delitem__(self, i):
        if self.rrset is None:
            raise IndexError
        del self.rrset[i]


class CacheStatistics:
    """Cache Statistics"""

    def __init__(self, hits=0, misses=0):
        self.hits = hits
        self.misses = misses

    def reset(self):
        self.hits = 0
        self.misses = 0

    def clone(self) -> "CacheStatistics":
        return CacheStatistics(self.hits, self.misses)


class CacheBase:
    def __init__(self):
        self.lock = threading.Lock()
        self.statistics = CacheStatistics()

    def reset_statistics(self) -> None:
        """Reset all statistics to zero."""
        with self.lock:
            self.statistics.reset()

    def hits(self) -> int:
        """How many hits has the cache had?"""
        with self.lock:
            return self.statistics.hits

    def misses(self) -> int:
        """How many misses has the cache had?"""
        with self.lock:
            return self.statistics.misses

    def get_statistics_snapshot(self) -> CacheStatistics:
        """Return a consistent snapshot of all the statistics.

        If running with multiple threads, it's better to take a
        snapshot than to call statistics methods such as hits() and
        misses() individually.
        """
        with self.lock:
            return self.statistics.clone()


CacheKey = Tuple[dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass]


class Cache(CacheBase):
    """Simple thread-safe DNS answer cache."""

    def __init__(self, cleaning_interval: float = 300.0):
        """*cleaning_interval*, a ``float`` is the number of seconds between
        periodic cleanings.
        """

        super().__init__()
        self.data: Dict[CacheKey, Answer] = {}
        self.cleaning_interval = cleaning_interval
        self.next_cleaning: float = time.time() + self.cleaning_interval

    def _maybe_clean(self) -> None:
        """Clean the cache if it's time to do so."""

        now = time.time()
        if self.next_cleaning <= now:
            keys_to_delete = []
            for (k, v) in self.data.items():
                if v.expiration <= now:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del self.data[k]
            now = time.time()
            self.next_cleaning = now + self.cleaning_interval

    def get(self, key: CacheKey) -> Optional[Answer]:
        """Get the answer associated with *key*.

        Returns None if no answer is cached for the key.

        *key*, a ``(dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass)``
        tuple whose values are the query name, rdtype, and rdclass respectively.

        Returns a ``dns.resolver.Answer`` or ``None``.
        """

        with self.lock:
            self._maybe_clean()
            v = self.data.get(key)
            if v is None or v.expiration <= time.time():
                self.statistics.misses += 1
                return None
            self.statistics.hits += 1
            return v

    def put(self, key: CacheKey, value: Answer) -> None:
        """Associate key and value in the cache.

        *key*, a ``(dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass)``
        tuple whose values are the query name, rdtype, and rdclass respectively.

        *value*, a ``dns.resolver.Answer``, the answer.
        """

        with self.lock:
            self._maybe_clean()
            self.data[key] = value

    def flush(self, key: Optional[CacheKey] = None) -> None:
        """Flush the cache.

        If *key* is not ``None``, only that item is flushed.  Otherwise the entire cache
        is flushed.

        *key*, a ``(dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass)``
        tuple whose values are the query name, rdtype, and rdclass respectively.
        """

        with self.lock:
            if key is not None:
                if key in self.data:
                    del self.data[key]
            else:
                self.data = {}
                self.next_cleaning = time.time() + self.cleaning_interval


class LRUCacheNode:
    """LRUCache node."""

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.hits = 0
        self.prev = self
        self.next = self

    def link_after(self, node):
        self.prev = node
        self.next = node.next
        node.next.prev = self
        node.next = self

    def unlink(self):
        self.next.prev = self.prev
        self.prev.next = self.next


class LRUCache(CacheBase):
    """Thread-safe, bounded, least-recently-used DNS answer cache.

    This cache is better than the simple cache (above) if you're
    running a web crawler or other process that does a lot of
    resolutions.  The LRUCache has a maximum number of nodes, and when
    it is full, the least-recently used node is removed to make space
    for a new one.
    """

    def __init__(self, max_size: int = 100000):
        """*max_size*, an ``int``, is the maximum number of nodes to cache;
        it must be greater than 0.
        """

        super().__init__()
        self.data: Dict[CacheKey, LRUCacheNode] = {}
        self.set_max_size(max_size)
        self.sentinel: LRUCacheNode = LRUCacheNode(None, None)
        self.sentinel.prev = self.sentinel
        self.sentinel.next = self.sentinel

    def set_max_size(self, max_size: int) -> None:
        if max_size < 1:
            max_size = 1
        self.max_size = max_size

    def get(self, key: CacheKey) -> Optional[Answer]:
        """Get the answer associated with *key*.

        Returns None if no answer is cached for the key.

        *key*, a ``(dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass)``
        tuple whose values are the query name, rdtype, and rdclass respectively.

        Returns a ``dns.resolver.Answer`` or ``None``.
        """

        with self.lock:
            node = self.data.get(key)
            if node is None:
                self.statistics.misses += 1
                return None
            # Unlink because we're either going to move the node to the front
            # of the LRU list or we're going to free it.
            node.unlink()
            if node.value.expiration <= time.time():
                del self.data[node.key]
                self.statistics.misses += 1
                return None
            node.link_after(self.sentinel)
            self.statistics.hits += 1
            node.hits += 1
            return node.value

    def get_hits_for_key(self, key: CacheKey) -> int:
        """Return the number of cache hits associated with the specified key."""
        with self.lock:
            node = self.data.get(key)
            if node is None or node.value.expiration <= time.time():
                return 0
            else:
                return node.hits

    def put(self, key: CacheKey, value: Answer) -> None:
        """Associate key and value in the cache.

        *key*, a ``(dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass)``
        tuple whose values are the query name, rdtype, and rdclass respectively.

        *value*, a ``dns.resolver.Answer``, the answer.
        """

        with self.lock:
            node = self.data.get(key)
            if node is not None:
                node.unlink()
                del self.data[node.key]
            while len(self.data) >= self.max_size:
                gnode = self.sentinel.prev
                gnode.unlink()
                del self.data[gnode.key]
            node = LRUCacheNode(key, value)
            node.link_after(self.sentinel)
            self.data[key] = node

    def flush(self, key: Optional[CacheKey] = None) -> None:
        """Flush the cache.

        If *key* is not ``None``, only that item is flushed.  Otherwise the entire cache
        is flushed.

        *key*, a ``(dns.name.Name, dns.rdatatype.RdataType, dns.rdataclass.RdataClass)``
        tuple whose values are the query name, rdtype, and rdclass respectively.
        """

        with self.lock:
            if key is not None:
                node = self.data.get(key)
                if node is not None:
                    node.unlink()
                    del self.data[node.key]
            else:
                gnode = self.sentinel.next
                while gnode != self.sentinel:
                    next = gnode.next
                    gnode.unlink()
                    gnode = next
                self.data = {}


class _Resolution:
    """Helper class for dns.resolver.Resolver.resolve().

    All of the "business logic" of resolution is encapsulated in this
    class, allowing us to have multiple resolve() implementations
    using different I/O schemes without copying all of the
    complicated logic.

    This class is a "friend" to dns.resolver.Resolver and manipulates
    resolver data structures directly.
    """

    def __init__(
        self,
        resolver: "BaseResolver",
        qname: Union[dns.name.Name, str],
        rdtype: Union[dns.rdatatype.RdataType, str],
        rdclass: Union[dns.rdataclass.RdataClass, str],
        tcp: bool,
        raise_on_no_answer: bool,
        search: Optional[bool],
    ):
        if isinstance(qname, str):
            qname = dns.name.from_text(qname, None)
        the_rdtype = dns.rdatatype.RdataType.make(rdtype)
        if dns.rdatatype.is_metatype(the_rdtype):
            raise NoMetaqueries
        the_rdclass = dns.rdataclass.RdataClass.make(rdclass)
        if dns.rdataclass.is_metaclass(the_rdclass):
            raise NoMetaqueries
        self.resolver = resolver
        self.qnames_to_try = resolver._get_qnames_to_try(qname, search)
        self.qnames = self.qnames_to_try[:]
        self.rdtype = the_rdtype
        self.rdclass = the_rdclass
        self.tcp = tcp
        self.raise_on_no_answer = raise_on_no_answer
        self.nxdomain_responses: Dict[dns.name.Name, dns.message.QueryMessage] = {}
        # Initialize other things to help analysis tools
        self.qname = dns.name.empty
        self.nameservers: List[str] = []
        self.current_nameservers: List[str] = []
        self.errors: List[ErrorTuple] = []
        self.nameserver: Optional[str] = None
        self.port = 0
        self.tcp_attempt = False
        self.retry_with_tcp = False
        self.request: Optional[dns.message.QueryMessage] = None
        self.backoff = 0.0

    def next_request(
        self,
    ) -> Tuple[Optional[dns.message.QueryMessage], Optional[Answer]]:
        """Get the next request to send, and check the cache.

        Returns a (request, answer) tuple.  At most one of request or
        answer will not be None.
        """

        # We return a tuple instead of Union[Message,Answer] as it lets
        # the caller avoid isinstance().

        while len(self.qnames) > 0:
            self.qname = self.qnames.pop(0)

            # Do we know the answer?
            if self.resolver.cache:
                answer = self.resolver.cache.get(
                    (self.qname, self.rdtype, self.rdclass)
                )
                if answer is not None:
                    if answer.rrset is None and self.raise_on_no_answer:
                        raise NoAnswer(response=answer.response)
                    else:
                        return (None, answer)
                answer = self.resolver.cache.get(
                    (self.qname, dns.rdatatype.ANY, self.rdclass)
                )
                if answer is not None and answer.response.rcode() == dns.rcode.NXDOMAIN:
                    # cached NXDOMAIN; record it and continue to next
                    # name.
                    self.nxdomain_responses[self.qname] = answer.response
                    continue

            # Build the request
            request = dns.message.make_query(self.qname, self.rdtype, self.rdclass)
            if self.resolver.keyname is not None:
                request.use_tsig(
                    self.resolver.keyring,
                    self.resolver.keyname,
                    algorithm=self.resolver.keyalgorithm,
                )
            request.use_edns(
                self.resolver.edns,
                self.resolver.ednsflags,
                self.resolver.payload,
                options=self.resolver.ednsoptions,
            )
            if self.resolver.flags is not None:
                request.flags = self.resolver.flags

            self.nameservers = self.resolver.nameservers[:]
            if self.resolver.rotate:
                random.shuffle(self.nameservers)
            self.current_nameservers = self.nameservers[:]
            self.errors = []
            self.nameserver = None
            self.tcp_attempt = False
            self.retry_with_tcp = False
            self.request = request
            self.backoff = 0.10

            return (request, None)

        #
        # We've tried everything and only gotten NXDOMAINs.  (We know
        # it's only NXDOMAINs as anything else would have returned
        # before now.)
        #
        raise NXDOMAIN(qnames=self.qnames_to_try, responses=self.nxdomain_responses)

    def next_nameserver(self) -> Tuple[str, int, bool, float]:
        if self.retry_with_tcp:
            assert self.nameserver is not None
            self.tcp_attempt = True
            self.retry_with_tcp = False
            return (self.nameserver, self.port, True, 0)

        backoff = 0.0
        if not self.current_nameservers:
            if len(self.nameservers) == 0:
                # Out of things to try!
                raise NoNameservers(request=self.request, errors=self.errors)
            self.current_nameservers = self.nameservers[:]
            backoff = self.backoff
            self.backoff = min(self.backoff * 2, 2)

        self.nameserver = self.current_nameservers.pop(0)
        self.port = self.resolver.nameserver_ports.get(
            self.nameserver, self.resolver.port
        )
        self.tcp_attempt = self.tcp
        return (self.nameserver, self.port, self.tcp_attempt, backoff)

    def query_result(
        self, response: Optional[dns.message.Message], ex: Optional[Exception]
    ) -> Tuple[Optional[Answer], bool]:
        #
        # returns an (answer: Answer, end_loop: bool) tuple.
        #
        assert self.nameserver is not None
        if ex:
            # Exception during I/O or from_wire()
            assert response is None
            self.errors.append(
                (self.nameserver, self.tcp_attempt, self.port, ex, response)
            )
            if (
                isinstance(ex, dns.exception.FormError)
                or isinstance(ex, EOFError)
                or isinstance(ex, OSError)
                or isinstance(ex, NotImplementedError)
            ):
                # This nameserver is no good, take it out of the mix.
                self.nameservers.remove(self.nameserver)
            elif isinstance(ex, dns.message.Truncated):
                if self.tcp_attempt:
                    # Truncation with TCP is no good!
                    self.nameservers.remove(self.nameserver)
                else:
                    self.retry_with_tcp = True
            return (None, False)
        # We got an answer!
        assert response is not None
        assert isinstance(response, dns.message.QueryMessage)
        rcode = response.rcode()
        if rcode == dns.rcode.NOERROR:
            try:
                answer = Answer(
                    self.qname,
                    self.rdtype,
                    self.rdclass,
                    response,
                    self.nameserver,
                    self.port,
                )
            except Exception as e:
                self.errors.append(
                    (self.nameserver, self.tcp_attempt, self.port, e, response)
                )
                # The nameserver is no good, take it out of the mix.
                self.nameservers.remove(self.nameserver)
                return (None, False)
            if self.resolver.cache:
                self.resolver.cache.put((self.qname, self.rdtype, self.rdclass), answer)
            if answer.rrset is None and self.raise_on_no_answer:
                raise NoAnswer(response=answer.response)
            return (answer, True)
        elif rcode == dns.rcode.NXDOMAIN:
            # Further validate the response by making an Answer, even
            # if we aren't going to cache it.
            try:
                answer = Answer(
                    self.qname, dns.rdatatype.ANY, dns.rdataclass.IN, response
                )
            except Exception as e:
                self.errors.append(
                    (self.nameserver, self.tcp_attempt, self.port, e, response)
                )
                # The nameserver is no good, take it out of the mix.
                self.nameservers.remove(self.nameserver)
                return (None, False)
            self.nxdomain_responses[self.qname] = response
            if self.resolver.cache:
                self.resolver.cache.put(
                    (self.qname, dns.rdatatype.ANY, self.rdclass), answer
                )
            # Make next_nameserver() return None, so caller breaks its
            # inner loop and calls next_request().
            return (None, True)
        elif rcode == dns.rcode.YXDOMAIN:
            yex = YXDOMAIN()
            self.errors.append(
                (self.nameserver, self.tcp_attempt, self.port, yex, response)
            )
            raise yex
        else:
            #
            # We got a response, but we're not happy with the
            # rcode in it.
            #
            if rcode != dns.rcode.SERVFAIL or not self.resolver.retry_servfail:
                self.nameservers.remove(self.nameserver)
            self.errors.append(
                (
                    self.nameserver,
                    self.tcp_attempt,
                    self.port,
                    dns.rcode.to_text(rcode),
                    response,
                )
            )
            return (None, False)


class BaseResolver:
    """DNS stub resolver."""

    # We initialize in reset()
    #
    # pylint: disable=attribute-defined-outside-init

    domain: dns.name.Name
    nameserver_ports: Dict[str, int]
    port: int
    search: List[dns.name.Name]
    use_search_by_default: bool
    timeout: float
    lifetime: float
    keyring: Optional[Any]
    keyname: Optional[Union[dns.name.Name, str]]
    keyalgorithm: Union[dns.name.Name, str]
    edns: int
    ednsflags: int
    ednsoptions: Optional[List[dns.edns.Option]]
    payload: int
    cache: Any
    flags: Optional[int]
    retry_servfail: bool
    rotate: bool
    ndots: Optional[int]

    def __init__(self, filename: str = "/etc/resolv.conf", configure: bool = True):
        """*filename*, a ``str`` or file object, specifying a file
        in standard /etc/resolv.conf format.  This parameter is meaningful
        only when *configure* is true and the platform is POSIX.

        *configure*, a ``bool``.  If True (the default), the resolver
        instance is configured in the normal fashion for the operating
        system the resolver is running on.  (I.e. by reading a
        /etc/resolv.conf file on POSIX systems and from the registry
        on Windows systems.)
        """

        self.reset()
        if configure:
            if sys.platform == "win32":
                self.read_registry()
            elif filename:
                self.read_resolv_conf(filename)

    def reset(self):
        """Reset all resolver configuration to the defaults."""

        self.domain = dns.name.Name(dns.name.from_text(socket.gethostname())[1:])
        if len(self.domain) == 0:
            self.domain = dns.name.root
        self.nameservers = []
        self.nameserver_ports = {}
        self.port = 53
        self.search = []
        self.use_search_by_default = False
        self.timeout = 2.0
        self.lifetime = 5.0
        self.keyring = None
        self.keyname = None
        self.keyalgorithm = dns.tsig.default_algorithm
        self.edns = -1
        self.ednsflags = 0
        self.ednsoptions = None
        self.payload = 0
        self.cache = None
        self.flags = None
        self.retry_servfail = False
        self.rotate = False
        self.ndots = None

    def read_resolv_conf(self, f: Any) -> None:
        """Process *f* as a file in the /etc/resolv.conf format.  If f is
        a ``str``, it is used as the name of the file to open; otherwise it
        is treated as the file itself.

        Interprets the following items:

        - nameserver - name server IP address

        - domain - local domain name

        - search - search list for host-name lookup

        - options - supported options are rotate, timeout, edns0, and ndots

        """

        if isinstance(f, str):
            try:
                cm: contextlib.AbstractContextManager = open(f)
            except OSError:
                # /etc/resolv.conf doesn't exist, can't be read, etc.
                raise NoResolverConfiguration(f"cannot open {f}")
        else:
            cm = contextlib.nullcontext(f)
        with cm as f:
            for l in f:
                if len(l) == 0 or l[0] == "#" or l[0] == ";":
                    continue
                tokens = l.split()

                # Any line containing less than 2 tokens is malformed
                if len(tokens) < 2:
                    continue

                if tokens[0] == "nameserver":
                    self.nameservers.append(tokens[1])
                elif tokens[0] == "domain":
                    self.domain = dns.name.from_text(tokens[1])
                    # domain and search are exclusive
                    self.search = []
                elif tokens[0] == "search":
                    # the last search wins
                    self.search = []
                    for suffix in tokens[1:]:
                        self.search.append(dns.name.from_text(suffix))
                    # We don't set domain as it is not used if
                    # len(self.search) > 0
                elif tokens[0] == "options":
                    for opt in tokens[1:]:
                        if opt == "rotate":
                            self.rotate = True
                        elif opt == "edns0":
                            self.use_edns()
                        elif "timeout" in opt:
                            try:
                                self.timeout = int(opt.split(":")[1])
                            except (ValueError, IndexError):
                                pass
                        elif "ndots" in opt:
                            try:
                                self.ndots = int(opt.split(":")[1])
                            except (ValueError, IndexError):
                                pass
        if len(self.nameservers) == 0:
            raise NoResolverConfiguration("no nameservers")

    def read_registry(self) -> None:
        """Extract resolver configuration from the Windows registry."""
        try:
            info = dns.win32util.get_dns_info()  # type: ignore
            if info.domain is not None:
                self.domain = info.domain
            self.nameservers = info.nameservers
            self.search = info.search
        except AttributeError:
            raise NotImplementedError

    def _compute_timeout(
        self,
        start: float,
        lifetime: Optional[float] = None,
        errors: Optional[List[ErrorTuple]] = None,
    ) -> float:
        lifetime = self.lifetime if lifetime is None else lifetime
        now = time.time()
        duration = now - start
        if errors is None:
            errors = []
        if duration < 0:
            if duration < -1:
                # Time going backwards is bad.  Just give up.
                raise LifetimeTimeout(timeout=duration, errors=errors)
            else:
                # Time went backwards, but only a little.  This can
                # happen, e.g. under vmware with older linux kernels.
                # Pretend it didn't happen.
                duration = 0
        if duration >= lifetime:
            raise LifetimeTimeout(timeout=duration, errors=errors)
        return min(lifetime - duration, self.timeout)

    def _get_qnames_to_try(
        self, qname: dns.name.Name, search: Optional[bool]
    ) -> List[dns.name.Name]:
        # This is a separate method so we can unit test the search
        # rules without requiring the Internet.
        if search is None:
            search = self.use_search_by_default
        qnames_to_try = []
        if qname.is_absolute():
            qnames_to_try.append(qname)
        else:
            abs_qname = qname.concatenate(dns.name.root)
            if search:
                if len(self.search) > 0:
                    # There is a search list, so use it exclusively
                    search_list = self.search[:]
                elif self.domain != dns.name.root and self.domain is not None:
                    # We have some notion of a domain that isn't the root, so
                    # use it as the search list.
                    search_list = [self.domain]
                else:
                    search_list = []
                # Figure out the effective ndots (default is 1)
                if self.ndots is None:
                    ndots = 1
                else:
                    ndots = self.ndots
                for suffix in search_list:
                    qnames_to_try.append(qname + suffix)
                if len(qname) > ndots:
                    # The name has at least ndots dots, so we should try an
                    # absolute query first.
                    qnames_to_try.insert(0, abs_qname)
                else:
                    # The name has less than ndots dots, so we should search
                    # first, then try the absolute name.
                    qnames_to_try.append(abs_qname)
            else:
                qnames_to_try.append(abs_qname)
        return qnames_to_try

    def use_tsig(
        self,
        keyring: Any,
        keyname: Optional[Union[dns.name.Name, str]] = None,
        algorithm: Union[dns.name.Name, str] = dns.tsig.default_algorithm,
    ) -> None:
        """Add a TSIG signature to each query.

        The parameters are passed to ``dns.message.Message.use_tsig()``;
        see its documentation for details.
        """

        self.keyring = keyring
        self.keyname = keyname
        self.keyalgorithm = algorithm

    def use_edns(
        self,
        edns: Optional[Union[int, bool]] = 0,
        ednsflags: int = 0,
        payload: int = dns.message.DEFAULT_EDNS_PAYLOAD,
        options: Optional[List[dns.edns.Option]] = None,
    ) -> None:
        """Configure EDNS behavior.

        *edns*, an ``int``, is the EDNS level to use.  Specifying
        ``None``, ``False``, or ``-1`` means "do not use EDNS", and in this case
        the other parameters are ignored.  Specifying ``True`` is
        equivalent to specifying 0, i.e. "use EDNS0".

        *ednsflags*, an ``int``, the EDNS flag values.

        *payload*, an ``int``, is the EDNS sender's payload field, which is the
        maximum size of UDP datagram the sender can handle.  I.e. how big
        a response to this message can be.

        *options*, a list of ``dns.edns.Option`` objects or ``None``, the EDNS
        options.
        """

        if edns is None or edns is False:
            edns = -1
        elif edns is True:
            edns = 0
        self.edns = edns
        self.ednsflags = ednsflags
        self.payload = payload
        self.ednsoptions = options

    def set_flags(self, flags: int) -> None:
        """Overrides the default flags with your own.

        *flags*, an ``int``, the message flags to use.
        """

        self.flags = flags

    @property
    def nameservers(self) -> List[str]:
        return self._nameservers

    @nameservers.setter
    def nameservers(self, nameservers: List[str]) -> None:
        """
        *nameservers*, a ``list`` of nameservers.

        Raises ``ValueError`` if *nameservers* is anything other than a
        ``list``.
        """
        if isinstance(nameservers, list):
            for nameserver in nameservers:
                if not dns.inet.is_address(nameserver):
                    try:
                        if urlparse(nameserver).scheme != "https":
                            raise NotImplementedError
                    except Exception:
                        raise ValueError(
                            f"nameserver {nameserver} is not an "
                            "IP address or valid https URL"
                        )
            self._nameservers = nameservers
        else:
            raise ValueError(
                "nameservers must be a list (not a {})".format(type(nameservers))
            )


class Resolver(BaseResolver):
    """DNS stub resolver."""

    def resolve(
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
    ) -> Answer:  # pylint: disable=arguments-differ
        """Query nameservers to find the answer to the question.

        The *qname*, *rdtype*, and *rdclass* parameters may be objects
        of the appropriate type, or strings that can be converted into objects
        of the appropriate type.

        *qname*, a ``dns.name.Name`` or ``str``, the query name.

        *rdtype*, an ``int`` or ``str``,  the query type.

        *rdclass*, an ``int`` or ``str``,  the query class.

        *tcp*, a ``bool``.  If ``True``, use TCP to make the query.

        *source*, a ``str`` or ``None``.  If not ``None``, bind to this IP
        address when making queries.

        *raise_on_no_answer*, a ``bool``.  If ``True``, raise
        ``dns.resolver.NoAnswer`` if there's no answer to the question.

        *source_port*, an ``int``, the port from which to send the message.

        *lifetime*, a ``float``, how many seconds a query should run
        before timing out.

        *search*, a ``bool`` or ``None``, determines whether the
        search list configured in the system's resolver configuration
        are used for relative names, and whether the resolver's domain
        may be added to relative names.  The default is ``None``,
        which causes the value of the resolver's
        ``use_search_by_default`` attribute to be used.

        Raises ``dns.resolver.LifetimeTimeout`` if no answers could be found
        in the specified lifetime.

        Raises ``dns.resolver.NXDOMAIN`` if the query name does not exist.

        Raises ``dns.resolver.YXDOMAIN`` if the query name is too long after
        DNAME substitution.

        Raises ``dns.resolver.NoAnswer`` if *raise_on_no_answer* is
        ``True`` and the query name exists but has no RRset of the
        desired type and class.

        Raises ``dns.resolver.NoNameservers`` if no non-broken
        nameservers are available to answer the question.

        Returns a ``dns.resolver.Answer`` instance.

        """

        resolution = _Resolution(
            self, qname, rdtype, rdclass, tcp, raise_on_no_answer, search
        )
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
                    time.sleep(backoff)
                timeout = self._compute_timeout(start, lifetime, resolution.errors)
                try:
                    if dns.inet.is_address(nameserver):
                        if tcp:
                            response = dns.query.tcp(
                                request,
                                nameserver,
                                timeout=timeout,
                                port=port,
                                source=source,
                                source_port=source_port,
                            )
                        else:
                            response = dns.query.udp(
                                request,
                                nameserver,
                                timeout=timeout,
                                port=port,
                                source=source,
                                source_port=source_port,
                                raise_on_truncation=True,
                            )
                    else:
                        response = dns.query.https(request, nameserver, timeout=timeout)
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

    def query(
        self,
        qname: Union[dns.name.Name, str],
        rdtype: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.A,
        rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
        tcp: bool = False,
        source: Optional[str] = None,
        raise_on_no_answer: bool = True,
        source_port: int = 0,
        lifetime: Optional[float] = None,
    ) -> Answer:  # pragma: no cover
        """Query nameservers to find the answer to the question.

        This method calls resolve() with ``search=True``, and is
        provided for backwards compatibility with prior versions of
        dnspython.  See the documentation for the resolve() method for
        further details.
        """
        warnings.warn(
            "please use dns.resolver.Resolver.resolve() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.resolve(
            qname,
            rdtype,
            rdclass,
            tcp,
            source,
            raise_on_no_answer,
            source_port,
            lifetime,
            True,
        )

    def resolve_address(self, ipaddr: str, *args: Any, **kwargs: Any) -> Answer:
        """Use a resolver to run a reverse query for PTR records.

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
        return self.resolve(
            dns.reversename.from_address(ipaddr), *args, **modified_kwargs
        )  # type: ignore[arg-type]

    # pylint: disable=redefined-outer-name

    def canonical_name(self, name: Union[dns.name.Name, str]) -> dns.name.Name:
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
            answer = self.resolve(name, raise_on_no_answer=False)
            canonical_name = answer.canonical_name
        except dns.resolver.NXDOMAIN as e:
            canonical_name = e.canonical_name
        return canonical_name

    # pylint: enable=redefined-outer-name


#: The default resolver.
default_resolver: Optional[Resolver] = None


def get_default_resolver() -> Resolver:
    """Get the default resolver, initializing it if necessary."""
    if default_resolver is None:
        reset_default_resolver()
    assert default_resolver is not None
    return default_resolver


def reset_default_resolver():
    """Re-initialize default resolver.

    Note that the resolver configuration (i.e. /etc/resolv.conf on UNIX
    systems) will be re-read immediately.
    """

    global default_resolver
    default_resolver = Resolver()


def resolve(
    qname: Union[dns.name.Name, str],
    rdtype: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.A,
    rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
    tcp: bool = False,
    source: Optional[str] = None,
    raise_on_no_answer: bool = True,
    source_port: int = 0,
    lifetime: Optional[float] = None,
    search: Optional[bool] = None,
) -> Answer:  # pragma: no cover

    """Query nameservers to find the answer to the question.

    This is a convenience function that uses the default resolver
    object to make the query.

    See ``dns.resolver.Resolver.resolve`` for more information on the
    parameters.
    """

    return get_default_resolver().resolve(
        qname,
        rdtype,
        rdclass,
        tcp,
        source,
        raise_on_no_answer,
        source_port,
        lifetime,
        search,
    )


def query(
    qname: Union[dns.name.Name, str],
    rdtype: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.A,
    rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
    tcp: bool = False,
    source: Optional[str] = None,
    raise_on_no_answer: bool = True,
    source_port: int = 0,
    lifetime: Optional[float] = None,
) -> Answer:  # pragma: no cover
    """Query nameservers to find the answer to the question.

    This method calls resolve() with ``search=True``, and is
    provided for backwards compatibility with prior versions of
    dnspython.  See the documentation for the resolve() method for
    further details.
    """
    warnings.warn(
        "please use dns.resolver.resolve() instead", DeprecationWarning, stacklevel=2
    )
    return resolve(
        qname,
        rdtype,
        rdclass,
        tcp,
        source,
        raise_on_no_answer,
        source_port,
        lifetime,
        True,
    )


def resolve_address(ipaddr: str, *args: Any, **kwargs: Any) -> Answer:
    """Use a resolver to run a reverse query for PTR records.

    See ``dns.resolver.Resolver.resolve_address`` for more information on the
    parameters.
    """

    return get_default_resolver().resolve_address(ipaddr, *args, **kwargs)


def canonical_name(name: Union[dns.name.Name, str]) -> dns.name.Name:
    """Determine the canonical name of *name*.

    See ``dns.resolver.Resolver.canonical_name`` for more information on the
    parameters and possible exceptions.
    """

    return get_default_resolver().canonical_name(name)


def zone_for_name(
    name: Union[dns.name.Name, str],
    rdclass: dns.rdataclass.RdataClass = dns.rdataclass.IN,
    tcp: bool = False,
    resolver: Optional[Resolver] = None,
    lifetime: Optional[float] = None,
) -> dns.name.Name:
    """Find the name of the zone which contains the specified name.

    *name*, an absolute ``dns.name.Name`` or ``str``, the query name.

    *rdclass*, an ``int``, the query class.

    *tcp*, a ``bool``.  If ``True``, use TCP to make the query.

    *resolver*, a ``dns.resolver.Resolver`` or ``None``, the resolver to use.
    If ``None``, the default, then the default resolver is used.

    *lifetime*, a ``float``, the total time to allow for the queries needed
    to determine the zone.  If ``None``, the default, then only the individual
    query limits of the resolver apply.

    Raises ``dns.resolver.NoRootSOA`` if there is no SOA RR at the DNS
    root.  (This is only likely to happen if you're using non-default
    root servers in your network and they are misconfigured.)

    Raises ``dns.resolver.LifetimeTimeout`` if the answer could not be
    found in the allotted lifetime.

    Returns a ``dns.name.Name``.
    """

    if isinstance(name, str):
        name = dns.name.from_text(name, dns.name.root)
    if resolver is None:
        resolver = get_default_resolver()
    if not name.is_absolute():
        raise NotAbsolute(name)
    start = time.time()
    expiration: Optional[float]
    if lifetime is not None:
        expiration = start + lifetime
    else:
        expiration = None
    while 1:
        try:
            rlifetime: Optional[float]
            if expiration:
                rlifetime = expiration - time.time()
                if rlifetime <= 0:
                    rlifetime = 0
            else:
                rlifetime = None
            answer = resolver.resolve(
                name, dns.rdatatype.SOA, rdclass, tcp, lifetime=rlifetime
            )
            assert answer.rrset is not None
            if answer.rrset.name == name:
                return name
            # otherwise we were CNAMEd or DNAMEd and need to look higher
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer) as e:
            if isinstance(e, dns.resolver.NXDOMAIN):
                response = e.responses().get(name)
            else:
                response = e.response()  # pylint: disable=no-value-for-parameter
            if response:
                for rrs in response.authority:
                    if rrs.rdtype == dns.rdatatype.SOA and rrs.rdclass == rdclass:
                        (nr, _, _) = rrs.name.fullcompare(name)
                        if nr == dns.name.NAMERELN_SUPERDOMAIN:
                            # We're doing a proper superdomain check as
                            # if the name were equal we ought to have gotten
                            # it in the answer section!  We are ignoring the
                            # possibility that the authority is insane and
                            # is including multiple SOA RRs for different
                            # authorities.
                            return rrs.name
            # we couldn't extract anything useful from the response (e.g. it's
            # a type 3 NXDOMAIN)
        try:
            name = name.parent()
        except dns.name.NoParent:
            raise NoRootSOA


#
# Support for overriding the system resolver for all python code in the
# running process.
#

_protocols_for_socktype = {
    socket.SOCK_DGRAM: [socket.SOL_UDP],
    socket.SOCK_STREAM: [socket.SOL_TCP],
}

_resolver = None
_original_getaddrinfo = socket.getaddrinfo
_original_getnameinfo = socket.getnameinfo
_original_getfqdn = socket.getfqdn
_original_gethostbyname = socket.gethostbyname
_original_gethostbyname_ex = socket.gethostbyname_ex
_original_gethostbyaddr = socket.gethostbyaddr


def _getaddrinfo(
    host=None, service=None, family=socket.AF_UNSPEC, socktype=0, proto=0, flags=0
):
    if flags & socket.AI_NUMERICHOST != 0:
        # Short circuit directly into the system's getaddrinfo().  We're
        # not adding any value in this case, and this avoids infinite loops
        # because dns.query.* needs to call getaddrinfo() for IPv6 scoping
        # reasons.  We will also do this short circuit below if we
        # discover that the host is an address literal.
        return _original_getaddrinfo(host, service, family, socktype, proto, flags)
    if flags & (socket.AI_ADDRCONFIG | socket.AI_V4MAPPED) != 0:
        # Not implemented.  We raise a gaierror as opposed to a
        # NotImplementedError as it helps callers handle errors more
        # appropriately.  [Issue #316]
        #
        # We raise EAI_FAIL as opposed to EAI_SYSTEM because there is
        # no EAI_SYSTEM on Windows [Issue #416].  We didn't go for
        # EAI_BADFLAGS as the flags aren't bad, we just don't
        # implement them.
        raise socket.gaierror(
            socket.EAI_FAIL, "Non-recoverable failure in name resolution"
        )
    if host is None and service is None:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    v6addrs = []
    v4addrs = []
    canonical_name = None  # pylint: disable=redefined-outer-name
    # Is host None or an address literal?  If so, use the system's
    # getaddrinfo().
    if host is None:
        return _original_getaddrinfo(host, service, family, socktype, proto, flags)
    try:
        # We don't care about the result of af_for_address(), we're just
        # calling it so it raises an exception if host is not an IPv4 or
        # IPv6 address.
        dns.inet.af_for_address(host)
        return _original_getaddrinfo(host, service, family, socktype, proto, flags)
    except Exception:
        pass
    # Something needs resolution!
    try:
        if family == socket.AF_INET6 or family == socket.AF_UNSPEC:
            v6 = _resolver.resolve(host, dns.rdatatype.AAAA, raise_on_no_answer=False)
            # Note that setting host ensures we query the same name
            # for A as we did for AAAA.  (This is just in case search lists
            # are active by default in the resolver configuration and
            # we might be talking to a server that says NXDOMAIN when it
            # wants to say NOERROR no data.
            host = v6.qname
            canonical_name = v6.canonical_name.to_text(True)
            if v6.rrset is not None:
                for rdata in v6.rrset:
                    v6addrs.append(rdata.address)
        if family == socket.AF_INET or family == socket.AF_UNSPEC:
            v4 = _resolver.resolve(host, dns.rdatatype.A, raise_on_no_answer=False)
            canonical_name = v4.canonical_name.to_text(True)
            if v4.rrset is not None:
                for rdata in v4.rrset:
                    v4addrs.append(rdata.address)
    except dns.resolver.NXDOMAIN:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    except Exception:
        # We raise EAI_AGAIN here as the failure may be temporary
        # (e.g. a timeout) and EAI_SYSTEM isn't defined on Windows.
        # [Issue #416]
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    port = None
    try:
        # Is it a port literal?
        if service is None:
            port = 0
        else:
            port = int(service)
    except Exception:
        if flags & socket.AI_NUMERICSERV == 0:
            try:
                port = socket.getservbyname(service)
            except Exception:
                pass
    if port is None:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    tuples = []
    if socktype == 0:
        socktypes = [socket.SOCK_DGRAM, socket.SOCK_STREAM]
    else:
        socktypes = [socktype]
    if flags & socket.AI_CANONNAME != 0:
        cname = canonical_name
    else:
        cname = ""
    if family == socket.AF_INET6 or family == socket.AF_UNSPEC:
        for addr in v6addrs:
            for socktype in socktypes:
                for proto in _protocols_for_socktype[socktype]:
                    tuples.append(
                        (socket.AF_INET6, socktype, proto, cname, (addr, port, 0, 0))
                    )
    if family == socket.AF_INET or family == socket.AF_UNSPEC:
        for addr in v4addrs:
            for socktype in socktypes:
                for proto in _protocols_for_socktype[socktype]:
                    tuples.append(
                        (socket.AF_INET, socktype, proto, cname, (addr, port))
                    )
    if len(tuples) == 0:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    return tuples


def _getnameinfo(sockaddr, flags=0):
    host = sockaddr[0]
    port = sockaddr[1]
    if len(sockaddr) == 4:
        scope = sockaddr[3]
        family = socket.AF_INET6
    else:
        scope = None
        family = socket.AF_INET
    tuples = _getaddrinfo(host, port, family, socket.SOCK_STREAM, socket.SOL_TCP, 0)
    if len(tuples) > 1:
        raise socket.error("sockaddr resolved to multiple addresses")
    addr = tuples[0][4][0]
    if flags & socket.NI_DGRAM:
        pname = "udp"
    else:
        pname = "tcp"
    qname = dns.reversename.from_address(addr)
    if flags & socket.NI_NUMERICHOST == 0:
        try:
            answer = _resolver.resolve(qname, "PTR")
            hostname = answer.rrset[0].target.to_text(True)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            if flags & socket.NI_NAMEREQD:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            hostname = addr
            if scope is not None:
                hostname += "%" + str(scope)
    else:
        hostname = addr
        if scope is not None:
            hostname += "%" + str(scope)
    if flags & socket.NI_NUMERICSERV:
        service = str(port)
    else:
        service = socket.getservbyport(port, pname)
    return (hostname, service)


def _getfqdn(name=None):
    if name is None:
        name = socket.gethostname()
    try:
        (name, _, _) = _gethostbyaddr(name)
        # Python's version checks aliases too, but our gethostbyname
        # ignores them, so we do so here as well.
    except Exception:
        pass
    return name


def _gethostbyname(name):
    return _gethostbyname_ex(name)[2][0]


def _gethostbyname_ex(name):
    aliases = []
    addresses = []
    tuples = _getaddrinfo(
        name, 0, socket.AF_INET, socket.SOCK_STREAM, socket.SOL_TCP, socket.AI_CANONNAME
    )
    canonical = tuples[0][3]
    for item in tuples:
        addresses.append(item[4][0])
    # XXX we just ignore aliases
    return (canonical, aliases, addresses)


def _gethostbyaddr(ip):
    try:
        dns.ipv6.inet_aton(ip)
        sockaddr = (ip, 80, 0, 0)
        family = socket.AF_INET6
    except Exception:
        try:
            dns.ipv4.inet_aton(ip)
        except Exception:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        sockaddr = (ip, 80)
        family = socket.AF_INET
    (name, _) = _getnameinfo(sockaddr, socket.NI_NAMEREQD)
    aliases = []
    addresses = []
    tuples = _getaddrinfo(
        name, 0, family, socket.SOCK_STREAM, socket.SOL_TCP, socket.AI_CANONNAME
    )
    canonical = tuples[0][3]
    # We only want to include an address from the tuples if it's the
    # same as the one we asked about.  We do this comparison in binary
    # to avoid any differences in text representations.
    bin_ip = dns.inet.inet_pton(family, ip)
    for item in tuples:
        addr = item[4][0]
        bin_addr = dns.inet.inet_pton(family, addr)
        if bin_ip == bin_addr:
            addresses.append(addr)
    # XXX we just ignore aliases
    return (canonical, aliases, addresses)


def override_system_resolver(resolver: Optional[Resolver] = None) -> None:
    """Override the system resolver routines in the socket module with
    versions which use dnspython's resolver.

    This can be useful in testing situations where you want to control
    the resolution behavior of python code without having to change
    the system's resolver settings (e.g. /etc/resolv.conf).

    The resolver to use may be specified; if it's not, the default
    resolver will be used.

    resolver, a ``dns.resolver.Resolver`` or ``None``, the resolver to use.
    """

    if resolver is None:
        resolver = get_default_resolver()
    global _resolver
    _resolver = resolver
    socket.getaddrinfo = _getaddrinfo
    socket.getnameinfo = _getnameinfo
    socket.getfqdn = _getfqdn
    socket.gethostbyname = _gethostbyname
    socket.gethostbyname_ex = _gethostbyname_ex
    socket.gethostbyaddr = _gethostbyaddr


def restore_system_resolver() -> None:
    """Undo the effects of prior override_system_resolver()."""

    global _resolver
    _resolver = None
    socket.getaddrinfo = _original_getaddrinfo
    socket.getnameinfo = _original_getnameinfo
    socket.getfqdn = _original_getfqdn
    socket.gethostbyname = _original_gethostbyname
    socket.gethostbyname_ex = _original_gethostbyname_ex
    socket.gethostbyaddr = _original_gethostbyaddr
