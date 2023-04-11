# From: https://github.com/ActiveState/code/blob/master/recipes/Python/576696_OrderedSet_with_Weakrefs/  # noqa
import typing as t
import weakref


class Link:
    """Representation of one item in a doubly-linked list."""

    __slots__ = ("prev", "next", "key", "__weakref__")
    prev: "Link"
    next: "Link"
    key: str


class OrderedSet(t.MutableSet[str]):
    """A set that remembers the order in which items were added."""

    # Big-O running times for all methods are the same as for regular sets.
    # The internal self.__map dictionary maps keys to links in a doubly linked
    # list. The circular doubly linked list starts and ends with a sentinel
    # element. The sentinel element never gets deleted (this simplifies the
    # algorithm). The prev/next links are weakref proxies (to prevent circular
    # references). Individual links are kept alive by the hard reference in
    # self.__map. Those hard references disappear when a key is deleted from
    # an OrderedSet.

    def __init__(self, iterable: t.Optional[t.Iterable[str]] = None):
        self.__root = root = Link()  # sentinel node for doubly linked list
        root.prev = root.next = root
        self.__map: t.MutableMapping[str, Link] = {}  # key --> link
        if iterable is not None:
            self |= iterable  # type: ignore

    def __len__(self) -> int:
        return len(self.__map)

    def __contains__(self, key: object) -> bool:
        return key in self.__map

    def add(self, key: str) -> None:
        # Store new key in a new link at the end of the linked list
        if key not in self.__map:
            self.__map[key] = link = Link()
            root = self.__root
            last = root.prev
            link.prev, link.next, link.key = last, root, key
            last.next = root.prev = weakref.proxy(link)

    def discard(self, key: str) -> None:
        # Remove an existing item using self.__map to find the link which is
        # then removed by updating the links in the predecessor and successors.
        if key in self.__map:
            link = self.__map.pop(key)
            link.prev.next = link.next
            link.next.prev = link.prev

    def __iter__(self) -> t.Generator[str, None, None]:
        # Traverse the linked list in order.
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next

    def __reversed__(self) -> t.Generator[str, None, None]:
        # Traverse the linked list in reverse order.
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.key
            curr = curr.prev

    def pop(self, last: bool = True) -> str:
        if not self:
            raise KeyError("set is empty")
        key = next(reversed(self)) if last else next(iter(self))
        self.discard(key)
        return key

    def __repr__(self) -> str:
        if not self:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}({list(self)!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        other = t.cast(t.Iterable[str], other)
        return not self.isdisjoint(other)
