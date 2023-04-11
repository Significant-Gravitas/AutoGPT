# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

from typing import Any, Callable, List, Optional, Tuple, Union

import collections

import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.serial
import dns.ttl


class TransactionManager:
    def reader(self) -> "Transaction":
        """Begin a read-only transaction."""
        raise NotImplementedError  # pragma: no cover

    def writer(self, replacement: bool = False) -> "Transaction":
        """Begin a writable transaction.

        *replacement*, a ``bool``.  If `True`, the content of the
        transaction completely replaces any prior content.  If False,
        the default, then the content of the transaction updates the
        existing content.
        """
        raise NotImplementedError  # pragma: no cover

    def origin_information(
        self,
    ) -> Tuple[Optional[dns.name.Name], bool, Optional[dns.name.Name]]:
        """Returns a tuple

            (absolute_origin, relativize, effective_origin)

        giving the absolute name of the default origin for any
        relative domain names, the "effective origin", and whether
        names should be relativized.  The "effective origin" is the
        absolute origin if relativize is False, and the empty name if
        relativize is true.  (The effective origin is provided even
        though it can be computed from the absolute_origin and
        relativize setting because it avoids a lot of code
        duplication.)

        If the returned names are `None`, then no origin information is
        available.

        This information is used by code working with transactions to
        allow it to coordinate relativization.  The transaction code
        itself takes what it gets (i.e. does not change name
        relativity).

        """
        raise NotImplementedError  # pragma: no cover

    def get_class(self) -> dns.rdataclass.RdataClass:
        """The class of the transaction manager."""
        raise NotImplementedError  # pragma: no cover

    def from_wire_origin(self) -> Optional[dns.name.Name]:
        """Origin to use in from_wire() calls."""
        (absolute_origin, relativize, _) = self.origin_information()
        if relativize:
            return absolute_origin
        else:
            return None


class DeleteNotExact(dns.exception.DNSException):
    """Existing data did not match data specified by an exact delete."""


class ReadOnly(dns.exception.DNSException):
    """Tried to write to a read-only transaction."""


class AlreadyEnded(dns.exception.DNSException):
    """Tried to use an already-ended transaction."""


def _ensure_immutable_rdataset(rdataset):
    if rdataset is None or isinstance(rdataset, dns.rdataset.ImmutableRdataset):
        return rdataset
    return dns.rdataset.ImmutableRdataset(rdataset)


def _ensure_immutable_node(node):
    if node is None or node.is_immutable():
        return node
    return dns.node.ImmutableNode(node)


CheckPutRdatasetType = Callable[
    ["Transaction", dns.name.Name, dns.rdataset.Rdataset], None
]
CheckDeleteRdatasetType = Callable[
    ["Transaction", dns.name.Name, dns.rdatatype.RdataType, dns.rdatatype.RdataType],
    None,
]
CheckDeleteNameType = Callable[["Transaction", dns.name.Name], None]


class Transaction:
    def __init__(
        self,
        manager: TransactionManager,
        replacement: bool = False,
        read_only: bool = False,
    ):
        self.manager = manager
        self.replacement = replacement
        self.read_only = read_only
        self._ended = False
        self._check_put_rdataset: List[CheckPutRdatasetType] = []
        self._check_delete_rdataset: List[CheckDeleteRdatasetType] = []
        self._check_delete_name: List[CheckDeleteNameType] = []

    #
    # This is the high level API
    #
    # Note that we currently use non-immutable types in the return type signature to
    # avoid covariance problems, e.g. if the caller has a List[Rdataset], mypy will be
    # unhappy if we return an ImmutableRdataset.

    def get(
        self,
        name: Optional[Union[dns.name.Name, str]],
        rdtype: Union[dns.rdatatype.RdataType, str],
        covers: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.NONE,
    ) -> dns.rdataset.Rdataset:
        """Return the rdataset associated with *name*, *rdtype*, and *covers*,
        or `None` if not found.

        Note that the returned rdataset is immutable.
        """
        self._check_ended()
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        rdtype = dns.rdatatype.RdataType.make(rdtype)
        covers = dns.rdatatype.RdataType.make(covers)
        rdataset = self._get_rdataset(name, rdtype, covers)
        return _ensure_immutable_rdataset(rdataset)

    def get_node(self, name: dns.name.Name) -> Optional[dns.node.Node]:
        """Return the node at *name*, if any.

        Returns an immutable node or ``None``.
        """
        return _ensure_immutable_node(self._get_node(name))

    def _check_read_only(self) -> None:
        if self.read_only:
            raise ReadOnly

    def add(self, *args: Any) -> None:
        """Add records.

        The arguments may be:

            - rrset

            - name, rdataset...

            - name, ttl, rdata...
        """
        self._check_ended()
        self._check_read_only()
        self._add(False, args)

    def replace(self, *args: Any) -> None:
        """Replace the existing rdataset at the name with the specified
        rdataset, or add the specified rdataset if there was no existing
        rdataset.

        The arguments may be:

            - rrset

            - name, rdataset...

            - name, ttl, rdata...

        Note that if you want to replace the entire node, you should do
        a delete of the name followed by one or more calls to add() or
        replace().
        """
        self._check_ended()
        self._check_read_only()
        self._add(True, args)

    def delete(self, *args: Any) -> None:
        """Delete records.

        It is not an error if some of the records are not in the existing
        set.

        The arguments may be:

            - rrset

            - name

            - name, rdataclass, rdatatype, [covers]

            - name, rdataset...

            - name, rdata...
        """
        self._check_ended()
        self._check_read_only()
        self._delete(False, args)

    def delete_exact(self, *args: Any) -> None:
        """Delete records.

        The arguments may be:

            - rrset

            - name

            - name, rdataclass, rdatatype, [covers]

            - name, rdataset...

            - name, rdata...

        Raises dns.transaction.DeleteNotExact if some of the records
        are not in the existing set.

        """
        self._check_ended()
        self._check_read_only()
        self._delete(True, args)

    def name_exists(self, name: Union[dns.name.Name, str]) -> bool:
        """Does the specified name exist?"""
        self._check_ended()
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        return self._name_exists(name)

    def update_serial(
        self,
        value: int = 1,
        relative: bool = True,
        name: dns.name.Name = dns.name.empty,
    ) -> None:
        """Update the serial number.

        *value*, an `int`, is an increment if *relative* is `True`, or the
        actual value to set if *relative* is `False`.

        Raises `KeyError` if there is no SOA rdataset at *name*.

        Raises `ValueError` if *value* is negative or if the increment is
        so large that it would cause the new serial to be less than the
        prior value.
        """
        self._check_ended()
        if value < 0:
            raise ValueError("negative update_serial() value")
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        rdataset = self._get_rdataset(name, dns.rdatatype.SOA, dns.rdatatype.NONE)
        if rdataset is None or len(rdataset) == 0:
            raise KeyError
        if relative:
            serial = dns.serial.Serial(rdataset[0].serial) + value
        else:
            serial = dns.serial.Serial(value)
        serial = serial.value  # convert back to int
        if serial == 0:
            serial = 1
        rdata = rdataset[0].replace(serial=serial)
        new_rdataset = dns.rdataset.from_rdata(rdataset.ttl, rdata)
        self.replace(name, new_rdataset)

    def __iter__(self):
        self._check_ended()
        return self._iterate_rdatasets()

    def changed(self) -> bool:
        """Has this transaction changed anything?

        For read-only transactions, the result is always `False`.

        For writable transactions, the result is `True` if at some time
        during the life of the transaction, the content was changed.
        """
        self._check_ended()
        return self._changed()

    def commit(self) -> None:
        """Commit the transaction.

        Normally transactions are used as context managers and commit
        or rollback automatically, but it may be done explicitly if needed.
        A ``dns.transaction.Ended`` exception will be raised if you try
        to use a transaction after it has been committed or rolled back.

        Raises an exception if the commit fails (in which case the transaction
        is also rolled back.
        """
        self._end(True)

    def rollback(self) -> None:
        """Rollback the transaction.

        Normally transactions are used as context managers and commit
        or rollback automatically, but it may be done explicitly if needed.
        A ``dns.transaction.AlreadyEnded`` exception will be raised if you try
        to use a transaction after it has been committed or rolled back.

        Rollback cannot otherwise fail.
        """
        self._end(False)

    def check_put_rdataset(self, check: CheckPutRdatasetType) -> None:
        """Call *check* before putting (storing) an rdataset.

        The function is called with the transaction, the name, and the rdataset.

        The check function may safely make non-mutating transaction method
        calls, but behavior is undefined if mutating transaction methods are
        called.  The check function should raise an exception if it objects to
        the put, and otherwise should return ``None``.
        """
        self._check_put_rdataset.append(check)

    def check_delete_rdataset(self, check: CheckDeleteRdatasetType) -> None:
        """Call *check* before deleting an rdataset.

        The function is called with the transaction, the name, the rdatatype,
        and the covered rdatatype.

        The check function may safely make non-mutating transaction method
        calls, but behavior is undefined if mutating transaction methods are
        called.  The check function should raise an exception if it objects to
        the put, and otherwise should return ``None``.
        """
        self._check_delete_rdataset.append(check)

    def check_delete_name(self, check: CheckDeleteNameType) -> None:
        """Call *check* before putting (storing) an rdataset.

        The function is called with the transaction and the name.

        The check function may safely make non-mutating transaction method
        calls, but behavior is undefined if mutating transaction methods are
        called.  The check function should raise an exception if it objects to
        the put, and otherwise should return ``None``.
        """
        self._check_delete_name.append(check)

    #
    # Helper methods
    #

    def _raise_if_not_empty(self, method, args):
        if len(args) != 0:
            raise TypeError(f"extra parameters to {method}")

    def _rdataset_from_args(self, method, deleting, args):
        try:
            arg = args.popleft()
            if isinstance(arg, dns.rrset.RRset):
                rdataset = arg.to_rdataset()
            elif isinstance(arg, dns.rdataset.Rdataset):
                rdataset = arg
            else:
                if deleting:
                    ttl = 0
                else:
                    if isinstance(arg, int):
                        ttl = arg
                        if ttl > dns.ttl.MAX_TTL:
                            raise ValueError(f"{method}: TTL value too big")
                    else:
                        raise TypeError(f"{method}: expected a TTL")
                    arg = args.popleft()
                if isinstance(arg, dns.rdata.Rdata):
                    rdataset = dns.rdataset.from_rdata(ttl, arg)
                else:
                    raise TypeError(f"{method}: expected an Rdata")
            return rdataset
        except IndexError:
            if deleting:
                return None
            else:
                # reraise
                raise TypeError(f"{method}: expected more arguments")

    def _add(self, replace, args):
        try:
            args = collections.deque(args)
            if replace:
                method = "replace()"
            else:
                method = "add()"
            arg = args.popleft()
            if isinstance(arg, str):
                arg = dns.name.from_text(arg, None)
            if isinstance(arg, dns.name.Name):
                name = arg
                rdataset = self._rdataset_from_args(method, False, args)
            elif isinstance(arg, dns.rrset.RRset):
                rrset = arg
                name = rrset.name
                # rrsets are also rdatasets, but they don't print the
                # same and can't be stored in nodes, so convert.
                rdataset = rrset.to_rdataset()
            else:
                raise TypeError(
                    f"{method} requires a name or RRset " + "as the first argument"
                )
            if rdataset.rdclass != self.manager.get_class():
                raise ValueError(f"{method} has objects of wrong RdataClass")
            if rdataset.rdtype == dns.rdatatype.SOA:
                (_, _, origin) = self._origin_information()
                if name != origin:
                    raise ValueError(f"{method} has non-origin SOA")
            self._raise_if_not_empty(method, args)
            if not replace:
                existing = self._get_rdataset(name, rdataset.rdtype, rdataset.covers)
                if existing is not None:
                    if isinstance(existing, dns.rdataset.ImmutableRdataset):
                        trds = dns.rdataset.Rdataset(
                            existing.rdclass, existing.rdtype, existing.covers
                        )
                        trds.update(existing)
                        existing = trds
                    rdataset = existing.union(rdataset)
            self._checked_put_rdataset(name, rdataset)
        except IndexError:
            raise TypeError(f"not enough parameters to {method}")

    def _delete(self, exact, args):
        try:
            args = collections.deque(args)
            if exact:
                method = "delete_exact()"
            else:
                method = "delete()"
            arg = args.popleft()
            if isinstance(arg, str):
                arg = dns.name.from_text(arg, None)
            if isinstance(arg, dns.name.Name):
                name = arg
                if len(args) > 0 and (
                    isinstance(args[0], int) or isinstance(args[0], str)
                ):
                    # deleting by type and (optionally) covers
                    rdtype = dns.rdatatype.RdataType.make(args.popleft())
                    if len(args) > 0:
                        covers = dns.rdatatype.RdataType.make(args.popleft())
                    else:
                        covers = dns.rdatatype.NONE
                    self._raise_if_not_empty(method, args)
                    existing = self._get_rdataset(name, rdtype, covers)
                    if existing is None:
                        if exact:
                            raise DeleteNotExact(f"{method}: missing rdataset")
                    else:
                        self._delete_rdataset(name, rdtype, covers)
                    return
                else:
                    rdataset = self._rdataset_from_args(method, True, args)
            elif isinstance(arg, dns.rrset.RRset):
                rdataset = arg  # rrsets are also rdatasets
                name = rdataset.name
            else:
                raise TypeError(
                    f"{method} requires a name or RRset " + "as the first argument"
                )
            self._raise_if_not_empty(method, args)
            if rdataset:
                if rdataset.rdclass != self.manager.get_class():
                    raise ValueError(f"{method} has objects of wrong RdataClass")
                existing = self._get_rdataset(name, rdataset.rdtype, rdataset.covers)
                if existing is not None:
                    if exact:
                        intersection = existing.intersection(rdataset)
                        if intersection != rdataset:
                            raise DeleteNotExact(f"{method}: missing rdatas")
                    rdataset = existing.difference(rdataset)
                    if len(rdataset) == 0:
                        self._checked_delete_rdataset(
                            name, rdataset.rdtype, rdataset.covers
                        )
                    else:
                        self._checked_put_rdataset(name, rdataset)
                elif exact:
                    raise DeleteNotExact(f"{method}: missing rdataset")
            else:
                if exact and not self._name_exists(name):
                    raise DeleteNotExact(f"{method}: name not known")
                self._checked_delete_name(name)
        except IndexError:
            raise TypeError(f"not enough parameters to {method}")

    def _check_ended(self):
        if self._ended:
            raise AlreadyEnded

    def _end(self, commit):
        self._check_ended()
        if self._ended:
            raise AlreadyEnded
        try:
            self._end_transaction(commit)
        finally:
            self._ended = True

    def _checked_put_rdataset(self, name, rdataset):
        for check in self._check_put_rdataset:
            check(self, name, rdataset)
        self._put_rdataset(name, rdataset)

    def _checked_delete_rdataset(self, name, rdtype, covers):
        for check in self._check_delete_rdataset:
            check(self, name, rdtype, covers)
        self._delete_rdataset(name, rdtype, covers)

    def _checked_delete_name(self, name):
        for check in self._check_delete_name:
            check(self, name)
        self._delete_name(name)

    #
    # Transactions are context managers.
    #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._ended:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        return False

    #
    # This is the low level API, which must be implemented by subclasses
    # of Transaction.
    #

    def _get_rdataset(self, name, rdtype, covers):
        """Return the rdataset associated with *name*, *rdtype*, and *covers*,
        or `None` if not found.
        """
        raise NotImplementedError  # pragma: no cover

    def _put_rdataset(self, name, rdataset):
        """Store the rdataset."""
        raise NotImplementedError  # pragma: no cover

    def _delete_name(self, name):
        """Delete all data associated with *name*.

        It is not an error if the name does not exist.
        """
        raise NotImplementedError  # pragma: no cover

    def _delete_rdataset(self, name, rdtype, covers):
        """Delete all data associated with *name*, *rdtype*, and *covers*.

        It is not an error if the rdataset does not exist.
        """
        raise NotImplementedError  # pragma: no cover

    def _name_exists(self, name):
        """Does name exist?

        Returns a bool.
        """
        raise NotImplementedError  # pragma: no cover

    def _changed(self):
        """Has this transaction changed anything?"""
        raise NotImplementedError  # pragma: no cover

    def _end_transaction(self, commit):
        """End the transaction.

        *commit*, a bool.  If ``True``, commit the transaction, otherwise
        roll it back.

        If committing and the commit fails, then roll back and raise an
        exception.
        """
        raise NotImplementedError  # pragma: no cover

    def _set_origin(self, origin):
        """Set the origin.

        This method is called when reading a possibly relativized
        source, and an origin setting operation occurs (e.g. $ORIGIN
        in a zone file).
        """
        raise NotImplementedError  # pragma: no cover

    def _iterate_rdatasets(self):
        """Return an iterator that yields (name, rdataset) tuples."""
        raise NotImplementedError  # pragma: no cover

    def _get_node(self, name):
        """Return the node at *name*, if any.

        Returns a node or ``None``.
        """
        raise NotImplementedError  # pragma: no cover

    #
    # Low-level API with a default implementation, in case a subclass needs
    # to override.
    #

    def _origin_information(self):
        # This is only used by _add()
        return self.manager.origin_information()
