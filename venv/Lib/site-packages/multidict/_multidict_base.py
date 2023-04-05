from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView


def _abc_itemsview_register(view_cls):
    ItemsView.register(view_cls)


def _abc_keysview_register(view_cls):
    KeysView.register(view_cls)


def _abc_valuesview_register(view_cls):
    ValuesView.register(view_cls)


def _viewbaseset_richcmp(view, other, op):
    if op == 0:  # <
        if not isinstance(other, Set):
            return NotImplemented
        return len(view) < len(other) and view <= other
    elif op == 1:  # <=
        if not isinstance(other, Set):
            return NotImplemented
        if len(view) > len(other):
            return False
        for elem in view:
            if elem not in other:
                return False
        return True
    elif op == 2:  # ==
        if not isinstance(other, Set):
            return NotImplemented
        return len(view) == len(other) and view <= other
    elif op == 3:  # !=
        return not view == other
    elif op == 4:  # >
        if not isinstance(other, Set):
            return NotImplemented
        return len(view) > len(other) and view >= other
    elif op == 5:  # >=
        if not isinstance(other, Set):
            return NotImplemented
        if len(view) < len(other):
            return False
        for elem in other:
            if elem not in view:
                return False
        return True


def _viewbaseset_and(view, other):
    if not isinstance(other, Iterable):
        return NotImplemented
    if isinstance(view, Set):
        view = set(iter(view))
    if isinstance(other, Set):
        other = set(iter(other))
    if not isinstance(other, Set):
        other = set(iter(other))
    return view & other


def _viewbaseset_or(view, other):
    if not isinstance(other, Iterable):
        return NotImplemented
    if isinstance(view, Set):
        view = set(iter(view))
    if isinstance(other, Set):
        other = set(iter(other))
    if not isinstance(other, Set):
        other = set(iter(other))
    return view | other


def _viewbaseset_sub(view, other):
    if not isinstance(other, Iterable):
        return NotImplemented
    if isinstance(view, Set):
        view = set(iter(view))
    if isinstance(other, Set):
        other = set(iter(other))
    if not isinstance(other, Set):
        other = set(iter(other))
    return view - other


def _viewbaseset_xor(view, other):
    if not isinstance(other, Iterable):
        return NotImplemented
    if isinstance(view, Set):
        view = set(iter(view))
    if isinstance(other, Set):
        other = set(iter(other))
    if not isinstance(other, Set):
        other = set(iter(other))
    return view ^ other


def _itemsview_isdisjoint(view, other):
    "Return True if two sets have a null intersection."
    for v in other:
        if v in view:
            return False
    return True


def _itemsview_repr(view):
    lst = []
    for k, v in view:
        lst.append("{!r}: {!r}".format(k, v))
    body = ", ".join(lst)
    return "{}({})".format(view.__class__.__name__, body)


def _keysview_isdisjoint(view, other):
    "Return True if two sets have a null intersection."
    for k in other:
        if k in view:
            return False
    return True


def _keysview_repr(view):
    lst = []
    for k in view:
        lst.append("{!r}".format(k))
    body = ", ".join(lst)
    return "{}({})".format(view.__class__.__name__, body)


def _valuesview_repr(view):
    lst = []
    for v in view:
        lst.append("{!r}".format(v))
    body = ", ".join(lst)
    return "{}({})".format(view.__class__.__name__, body)


def _mdrepr(md):
    lst = []
    for k, v in md.items():
        lst.append("'{}': {!r}".format(k, v))
    body = ", ".join(lst)
    return "<{}({})>".format(md.__class__.__name__, body)
