# SPDX-License-Identifier: MIT


import functools
import types

from ._make import _make_ne


_operation_names = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def cmp_using(
    eq=None,
    lt=None,
    le=None,
    gt=None,
    ge=None,
    require_same_type=True,
    class_name="Comparable",
):
    """
    Create a class that can be passed into `attr.ib`'s ``eq``, ``order``, and
    ``cmp`` arguments to customize field comparison.

    The resulting class will have a full set of ordering methods if
    at least one of ``{lt, le, gt, ge}`` and ``eq``  are provided.

    :param Optional[callable] eq: `callable` used to evaluate equality
        of two objects.
    :param Optional[callable] lt: `callable` used to evaluate whether
        one object is less than another object.
    :param Optional[callable] le: `callable` used to evaluate whether
        one object is less than or equal to another object.
    :param Optional[callable] gt: `callable` used to evaluate whether
        one object is greater than another object.
    :param Optional[callable] ge: `callable` used to evaluate whether
        one object is greater than or equal to another object.

    :param bool require_same_type: When `True`, equality and ordering methods
        will return `NotImplemented` if objects are not of the same type.

    :param Optional[str] class_name: Name of class. Defaults to 'Comparable'.

    See `comparison` for more details.

    .. versionadded:: 21.1.0
    """

    body = {
        "__slots__": ["value"],
        "__init__": _make_init(),
        "_requirements": [],
        "_is_comparable_to": _is_comparable_to,
    }

    # Add operations.
    num_order_functions = 0
    has_eq_function = False

    if eq is not None:
        has_eq_function = True
        body["__eq__"] = _make_operator("eq", eq)
        body["__ne__"] = _make_ne()

    if lt is not None:
        num_order_functions += 1
        body["__lt__"] = _make_operator("lt", lt)

    if le is not None:
        num_order_functions += 1
        body["__le__"] = _make_operator("le", le)

    if gt is not None:
        num_order_functions += 1
        body["__gt__"] = _make_operator("gt", gt)

    if ge is not None:
        num_order_functions += 1
        body["__ge__"] = _make_operator("ge", ge)

    type_ = types.new_class(
        class_name, (object,), {}, lambda ns: ns.update(body)
    )

    # Add same type requirement.
    if require_same_type:
        type_._requirements.append(_check_same_type)

    # Add total ordering if at least one operation was defined.
    if 0 < num_order_functions < 4:
        if not has_eq_function:
            # functools.total_ordering requires __eq__ to be defined,
            # so raise early error here to keep a nice stack.
            raise ValueError(
                "eq must be define is order to complete ordering from "
                "lt, le, gt, ge."
            )
        type_ = functools.total_ordering(type_)

    return type_


def _make_init():
    """
    Create __init__ method.
    """

    def __init__(self, value):
        """
        Initialize object with *value*.
        """
        self.value = value

    return __init__


def _make_operator(name, func):
    """
    Create operator method.
    """

    def method(self, other):
        if not self._is_comparable_to(other):
            return NotImplemented

        result = func(self.value, other.value)
        if result is NotImplemented:
            return NotImplemented

        return result

    method.__name__ = f"__{name}__"
    method.__doc__ = (
        f"Return a {_operation_names[name]} b.  Computed by attrs."
    )

    return method


def _is_comparable_to(self, other):
    """
    Check whether `other` is comparable to `self`.
    """
    for func in self._requirements:
        if not func(self, other):
            return False
    return True


def _check_same_type(self, other):
    """
    Return True if *self* and *other* are of the same type, False otherwise.
    """
    return other.value.__class__ is self.value.__class__
