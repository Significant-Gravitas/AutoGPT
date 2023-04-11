# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

"""DNS Versioned Zones."""

from typing import Callable, Deque, Optional, Set, Union

import collections
import threading

import dns.exception
import dns.immutable
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.zone


class UseTransaction(dns.exception.DNSException):
    """To alter a versioned zone, use a transaction."""


# Backwards compatibility
Node = dns.zone.VersionedNode
ImmutableNode = dns.zone.ImmutableVersionedNode
Version = dns.zone.Version
WritableVersion = dns.zone.WritableVersion
ImmutableVersion = dns.zone.ImmutableVersion
Transaction = dns.zone.Transaction


class Zone(dns.zone.Zone):  # lgtm[py/missing-equals]

    __slots__ = [
        "_versions",
        "_versions_lock",
        "_write_txn",
        "_write_waiters",
        "_write_event",
        "_pruning_policy",
        "_readers",
    ]

    node_factory = Node

    def __init__(
        self,
        origin: Optional[Union[dns.name.Name, str]],
        rdclass: dns.rdataclass.RdataClass = dns.rdataclass.IN,
        relativize: bool = True,
        pruning_policy: Optional[Callable[["Zone", Version], Optional[bool]]] = None,
    ):
        """Initialize a versioned zone object.

        *origin* is the origin of the zone.  It may be a ``dns.name.Name``,
        a ``str``, or ``None``.  If ``None``, then the zone's origin will
        be set by the first ``$ORIGIN`` line in a zone file.

        *rdclass*, an ``int``, the zone's rdata class; the default is class IN.

        *relativize*, a ``bool``, determine's whether domain names are
        relativized to the zone's origin.  The default is ``True``.

        *pruning policy*, a function taking a ``Zone`` and a ``Version`` and returning
        a ``bool``, or ``None``.  Should the version be pruned?  If ``None``,
        the default policy, which retains one version is used.
        """
        super().__init__(origin, rdclass, relativize)
        self._versions: Deque[Version] = collections.deque()
        self._version_lock = threading.Lock()
        if pruning_policy is None:
            self._pruning_policy = self._default_pruning_policy
        else:
            self._pruning_policy = pruning_policy
        self._write_txn: Optional[Transaction] = None
        self._write_event: Optional[threading.Event] = None
        self._write_waiters: Deque[threading.Event] = collections.deque()
        self._readers: Set[Transaction] = set()
        self._commit_version_unlocked(
            None, WritableVersion(self, replacement=True), origin
        )

    def reader(
        self, id: Optional[int] = None, serial: Optional[int] = None
    ) -> Transaction:  # pylint: disable=arguments-differ
        if id is not None and serial is not None:
            raise ValueError("cannot specify both id and serial")
        with self._version_lock:
            if id is not None:
                version = None
                for v in reversed(self._versions):
                    if v.id == id:
                        version = v
                        break
                if version is None:
                    raise KeyError("version not found")
            elif serial is not None:
                if self.relativize:
                    oname = dns.name.empty
                else:
                    assert self.origin is not None
                    oname = self.origin
                version = None
                for v in reversed(self._versions):
                    n = v.nodes.get(oname)
                    if n:
                        rds = n.get_rdataset(self.rdclass, dns.rdatatype.SOA)
                        if rds and rds[0].serial == serial:
                            version = v
                            break
                if version is None:
                    raise KeyError("serial not found")
            else:
                version = self._versions[-1]
            txn = Transaction(self, False, version)
            self._readers.add(txn)
            return txn

    def writer(self, replacement: bool = False) -> Transaction:
        event = None
        while True:
            with self._version_lock:
                # Checking event == self._write_event ensures that either
                # no one was waiting before we got lucky and found no write
                # txn, or we were the one who was waiting and got woken up.
                # This prevents "taking cuts" when creating a write txn.
                if self._write_txn is None and event == self._write_event:
                    # Creating the transaction defers version setup
                    # (i.e.  copying the nodes dictionary) until we
                    # give up the lock, so that we hold the lock as
                    # short a time as possible.  This is why we call
                    # _setup_version() below.
                    self._write_txn = Transaction(
                        self, replacement, make_immutable=True
                    )
                    # give up our exclusive right to make a Transaction
                    self._write_event = None
                    break
                # Someone else is writing already, so we will have to
                # wait, but we want to do the actual wait outside the
                # lock.
                event = threading.Event()
                self._write_waiters.append(event)
            # wait (note we gave up the lock!)
            #
            # We only wake one sleeper at a time, so it's important
            # that no event waiter can exit this method (e.g. via
            # cancellation) without returning a transaction or waking
            # someone else up.
            #
            # This is not a problem with Threading module threads as
            # they cannot be canceled, but could be an issue with trio
            # or curio tasks when we do the async version of writer().
            # I.e. we'd need to do something like:
            #
            # try:
            #     event.wait()
            # except trio.Cancelled:
            #     with self._version_lock:
            #         self._maybe_wakeup_one_waiter_unlocked()
            #     raise
            #
            event.wait()
        # Do the deferred version setup.
        self._write_txn._setup_version()
        return self._write_txn

    def _maybe_wakeup_one_waiter_unlocked(self):
        if len(self._write_waiters) > 0:
            self._write_event = self._write_waiters.popleft()
            self._write_event.set()

    # pylint: disable=unused-argument
    def _default_pruning_policy(self, zone, version):
        return True

    # pylint: enable=unused-argument

    def _prune_versions_unlocked(self):
        assert len(self._versions) > 0
        # Don't ever prune a version greater than or equal to one that
        # a reader has open.  This pins versions in memory while the
        # reader is open, and importantly lets the reader open a txn on
        # a successor version (e.g. if generating an IXFR).
        #
        # Note our definition of least_kept also ensures we do not try to
        # delete the greatest version.
        if len(self._readers) > 0:
            least_kept = min(txn.version.id for txn in self._readers)
        else:
            least_kept = self._versions[-1].id
        while self._versions[0].id < least_kept and self._pruning_policy(
            self, self._versions[0]
        ):
            self._versions.popleft()

    def set_max_versions(self, max_versions: Optional[int]) -> None:
        """Set a pruning policy that retains up to the specified number
        of versions
        """
        if max_versions is not None and max_versions < 1:
            raise ValueError("max versions must be at least 1")
        if max_versions is None:

            def policy(zone, _):  # pylint: disable=unused-argument
                return False

        else:

            def policy(zone, _):
                return len(zone._versions) > max_versions

        self.set_pruning_policy(policy)

    def set_pruning_policy(
        self, policy: Optional[Callable[["Zone", Version], Optional[bool]]]
    ) -> None:
        """Set the pruning policy for the zone.

        The *policy* function takes a `Version` and returns `True` if
        the version should be pruned, and `False` otherwise.  `None`
        may also be specified for policy, in which case the default policy
        is used.

        Pruning checking proceeds from the least version and the first
        time the function returns `False`, the checking stops.  I.e. the
        retained versions are always a consecutive sequence.
        """
        if policy is None:
            policy = self._default_pruning_policy
        with self._version_lock:
            self._pruning_policy = policy
            self._prune_versions_unlocked()

    def _end_read(self, txn):
        with self._version_lock:
            self._readers.remove(txn)
            self._prune_versions_unlocked()

    def _end_write_unlocked(self, txn):
        assert self._write_txn == txn
        self._write_txn = None
        self._maybe_wakeup_one_waiter_unlocked()

    def _end_write(self, txn):
        with self._version_lock:
            self._end_write_unlocked(txn)

    def _commit_version_unlocked(self, txn, version, origin):
        self._versions.append(version)
        self._prune_versions_unlocked()
        self.nodes = version.nodes
        if self.origin is None:
            self.origin = origin
        # txn can be None in __init__ when we make the empty version.
        if txn is not None:
            self._end_write_unlocked(txn)

    def _commit_version(self, txn, version, origin):
        with self._version_lock:
            self._commit_version_unlocked(txn, version, origin)

    def _get_next_version_id(self):
        if len(self._versions) > 0:
            id = self._versions[-1].id + 1
        else:
            id = 1
        return id

    def find_node(
        self, name: Union[dns.name.Name, str], create: bool = False
    ) -> dns.node.Node:
        if create:
            raise UseTransaction
        return super().find_node(name)

    def delete_node(self, name: Union[dns.name.Name, str]) -> None:
        raise UseTransaction

    def find_rdataset(
        self,
        name: Union[dns.name.Name, str],
        rdtype: Union[dns.rdatatype.RdataType, str],
        covers: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.NONE,
        create: bool = False,
    ) -> dns.rdataset.Rdataset:
        if create:
            raise UseTransaction
        rdataset = super().find_rdataset(name, rdtype, covers)
        return dns.rdataset.ImmutableRdataset(rdataset)

    def get_rdataset(
        self,
        name: Union[dns.name.Name, str],
        rdtype: Union[dns.rdatatype.RdataType, str],
        covers: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.NONE,
        create: bool = False,
    ) -> Optional[dns.rdataset.Rdataset]:
        if create:
            raise UseTransaction
        rdataset = super().get_rdataset(name, rdtype, covers)
        if rdataset is not None:
            return dns.rdataset.ImmutableRdataset(rdataset)
        else:
            return None

    def delete_rdataset(
        self,
        name: Union[dns.name.Name, str],
        rdtype: Union[dns.rdatatype.RdataType, str],
        covers: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.NONE,
    ) -> None:
        raise UseTransaction

    def replace_rdataset(
        self, name: Union[dns.name.Name, str], replacement: dns.rdataset.Rdataset
    ) -> None:
        raise UseTransaction
