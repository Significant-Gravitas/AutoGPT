# SPDX-License-Identifier: MIT


import copy

from ._make import NOTHING, _obj_setattr, fields
from .exceptions import AttrsAttributeNotFoundError


def asdict(
    inst,
    recurse=True,
    filter=None,
    dict_factory=dict,
    retain_collection_types=False,
    value_serializer=None,
):
    """
    Return the ``attrs`` attribute values of *inst* as a dict.

    Optionally recurse into other ``attrs``-decorated classes.

    :param inst: Instance of an ``attrs``-decorated class.
    :param bool recurse: Recurse into classes that are also
        ``attrs``-decorated.
    :param callable filter: A callable whose return code determines whether an
        attribute or element is included (``True``) or dropped (``False``).  Is
        called with the `attrs.Attribute` as the first argument and the
        value as the second argument.
    :param callable dict_factory: A callable to produce dictionaries from.  For
        example, to produce ordered dictionaries instead of normal Python
        dictionaries, pass in ``collections.OrderedDict``.
    :param bool retain_collection_types: Do not convert to ``list`` when
        encountering an attribute whose type is ``tuple`` or ``set``.  Only
        meaningful if ``recurse`` is ``True``.
    :param Optional[callable] value_serializer: A hook that is called for every
        attribute or dict key/value.  It receives the current instance, field
        and value and must return the (updated) value.  The hook is run *after*
        the optional *filter* has been applied.

    :rtype: return type of *dict_factory*

    :raise attr.exceptions.NotAnAttrsClassError: If *cls* is not an ``attrs``
        class.

    ..  versionadded:: 16.0.0 *dict_factory*
    ..  versionadded:: 16.1.0 *retain_collection_types*
    ..  versionadded:: 20.3.0 *value_serializer*
    ..  versionadded:: 21.3.0 If a dict has a collection for a key, it is
        serialized as a tuple.
    """
    attrs = fields(inst.__class__)
    rv = dict_factory()
    for a in attrs:
        v = getattr(inst, a.name)
        if filter is not None and not filter(a, v):
            continue

        if value_serializer is not None:
            v = value_serializer(inst, a, v)

        if recurse is True:
            if has(v.__class__):
                rv[a.name] = asdict(
                    v,
                    recurse=True,
                    filter=filter,
                    dict_factory=dict_factory,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                )
            elif isinstance(v, (tuple, list, set, frozenset)):
                cf = v.__class__ if retain_collection_types is True else list
                rv[a.name] = cf(
                    [
                        _asdict_anything(
                            i,
                            is_key=False,
                            filter=filter,
                            dict_factory=dict_factory,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        )
                        for i in v
                    ]
                )
            elif isinstance(v, dict):
                df = dict_factory
                rv[a.name] = df(
                    (
                        _asdict_anything(
                            kk,
                            is_key=True,
                            filter=filter,
                            dict_factory=df,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        ),
                        _asdict_anything(
                            vv,
                            is_key=False,
                            filter=filter,
                            dict_factory=df,
                            retain_collection_types=retain_collection_types,
                            value_serializer=value_serializer,
                        ),
                    )
                    for kk, vv in v.items()
                )
            else:
                rv[a.name] = v
        else:
            rv[a.name] = v
    return rv


def _asdict_anything(
    val,
    is_key,
    filter,
    dict_factory,
    retain_collection_types,
    value_serializer,
):
    """
    ``asdict`` only works on attrs instances, this works on anything.
    """
    if getattr(val.__class__, "__attrs_attrs__", None) is not None:
        # Attrs class.
        rv = asdict(
            val,
            recurse=True,
            filter=filter,
            dict_factory=dict_factory,
            retain_collection_types=retain_collection_types,
            value_serializer=value_serializer,
        )
    elif isinstance(val, (tuple, list, set, frozenset)):
        if retain_collection_types is True:
            cf = val.__class__
        elif is_key:
            cf = tuple
        else:
            cf = list

        rv = cf(
            [
                _asdict_anything(
                    i,
                    is_key=False,
                    filter=filter,
                    dict_factory=dict_factory,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                )
                for i in val
            ]
        )
    elif isinstance(val, dict):
        df = dict_factory
        rv = df(
            (
                _asdict_anything(
                    kk,
                    is_key=True,
                    filter=filter,
                    dict_factory=df,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                ),
                _asdict_anything(
                    vv,
                    is_key=False,
                    filter=filter,
                    dict_factory=df,
                    retain_collection_types=retain_collection_types,
                    value_serializer=value_serializer,
                ),
            )
            for kk, vv in val.items()
        )
    else:
        rv = val
        if value_serializer is not None:
            rv = value_serializer(None, None, rv)

    return rv


def astuple(
    inst,
    recurse=True,
    filter=None,
    tuple_factory=tuple,
    retain_collection_types=False,
):
    """
    Return the ``attrs`` attribute values of *inst* as a tuple.

    Optionally recurse into other ``attrs``-decorated classes.

    :param inst: Instance of an ``attrs``-decorated class.
    :param bool recurse: Recurse into classes that are also
        ``attrs``-decorated.
    :param callable filter: A callable whose return code determines whether an
        attribute or element is included (``True``) or dropped (``False``).  Is
        called with the `attrs.Attribute` as the first argument and the
        value as the second argument.
    :param callable tuple_factory: A callable to produce tuples from.  For
        example, to produce lists instead of tuples.
    :param bool retain_collection_types: Do not convert to ``list``
        or ``dict`` when encountering an attribute which type is
        ``tuple``, ``dict`` or ``set``.  Only meaningful if ``recurse`` is
        ``True``.

    :rtype: return type of *tuple_factory*

    :raise attr.exceptions.NotAnAttrsClassError: If *cls* is not an ``attrs``
        class.

    ..  versionadded:: 16.2.0
    """
    attrs = fields(inst.__class__)
    rv = []
    retain = retain_collection_types  # Very long. :/
    for a in attrs:
        v = getattr(inst, a.name)
        if filter is not None and not filter(a, v):
            continue
        if recurse is True:
            if has(v.__class__):
                rv.append(
                    astuple(
                        v,
                        recurse=True,
                        filter=filter,
                        tuple_factory=tuple_factory,
                        retain_collection_types=retain,
                    )
                )
            elif isinstance(v, (tuple, list, set, frozenset)):
                cf = v.__class__ if retain is True else list
                rv.append(
                    cf(
                        [
                            astuple(
                                j,
                                recurse=True,
                                filter=filter,
                                tuple_factory=tuple_factory,
                                retain_collection_types=retain,
                            )
                            if has(j.__class__)
                            else j
                            for j in v
                        ]
                    )
                )
            elif isinstance(v, dict):
                df = v.__class__ if retain is True else dict
                rv.append(
                    df(
                        (
                            astuple(
                                kk,
                                tuple_factory=tuple_factory,
                                retain_collection_types=retain,
                            )
                            if has(kk.__class__)
                            else kk,
                            astuple(
                                vv,
                                tuple_factory=tuple_factory,
                                retain_collection_types=retain,
                            )
                            if has(vv.__class__)
                            else vv,
                        )
                        for kk, vv in v.items()
                    )
                )
            else:
                rv.append(v)
        else:
            rv.append(v)

    return rv if tuple_factory is list else tuple_factory(rv)


def has(cls):
    """
    Check whether *cls* is a class with ``attrs`` attributes.

    :param type cls: Class to introspect.
    :raise TypeError: If *cls* is not a class.

    :rtype: bool
    """
    return getattr(cls, "__attrs_attrs__", None) is not None


def assoc(inst, **changes):
    """
    Copy *inst* and apply *changes*.

    :param inst: Instance of a class with ``attrs`` attributes.
    :param changes: Keyword changes in the new copy.

    :return: A copy of inst with *changes* incorporated.

    :raise attr.exceptions.AttrsAttributeNotFoundError: If *attr_name* couldn't
        be found on *cls*.
    :raise attr.exceptions.NotAnAttrsClassError: If *cls* is not an ``attrs``
        class.

    ..  deprecated:: 17.1.0
        Use `attrs.evolve` instead if you can.
        This function will not be removed du to the slightly different approach
        compared to `attrs.evolve`.
    """
    import warnings

    warnings.warn(
        "assoc is deprecated and will be removed after 2018/01.",
        DeprecationWarning,
        stacklevel=2,
    )
    new = copy.copy(inst)
    attrs = fields(inst.__class__)
    for k, v in changes.items():
        a = getattr(attrs, k, NOTHING)
        if a is NOTHING:
            raise AttrsAttributeNotFoundError(
                f"{k} is not an attrs attribute on {new.__class__}."
            )
        _obj_setattr(new, k, v)
    return new


def evolve(inst, **changes):
    """
    Create a new instance, based on *inst* with *changes* applied.

    :param inst: Instance of a class with ``attrs`` attributes.
    :param changes: Keyword changes in the new copy.

    :return: A copy of inst with *changes* incorporated.

    :raise TypeError: If *attr_name* couldn't be found in the class
        ``__init__``.
    :raise attr.exceptions.NotAnAttrsClassError: If *cls* is not an ``attrs``
        class.

    ..  versionadded:: 17.1.0
    """
    cls = inst.__class__
    attrs = fields(cls)
    for a in attrs:
        if not a.init:
            continue
        attr_name = a.name  # To deal with private attributes.
        init_name = a.alias
        if init_name not in changes:
            changes[init_name] = getattr(inst, attr_name)

    return cls(**changes)


def resolve_types(cls, globalns=None, localns=None, attribs=None):
    """
    Resolve any strings and forward annotations in type annotations.

    This is only required if you need concrete types in `Attribute`'s *type*
    field. In other words, you don't need to resolve your types if you only
    use them for static type checking.

    With no arguments, names will be looked up in the module in which the class
    was created. If this is not what you want, e.g. if the name only exists
    inside a method, you may pass *globalns* or *localns* to specify other
    dictionaries in which to look up these names. See the docs of
    `typing.get_type_hints` for more details.

    :param type cls: Class to resolve.
    :param Optional[dict] globalns: Dictionary containing global variables.
    :param Optional[dict] localns: Dictionary containing local variables.
    :param Optional[list] attribs: List of attribs for the given class.
        This is necessary when calling from inside a ``field_transformer``
        since *cls* is not an ``attrs`` class yet.

    :raise TypeError: If *cls* is not a class.
    :raise attr.exceptions.NotAnAttrsClassError: If *cls* is not an ``attrs``
        class and you didn't pass any attribs.
    :raise NameError: If types cannot be resolved because of missing variables.

    :returns: *cls* so you can use this function also as a class decorator.
        Please note that you have to apply it **after** `attrs.define`. That
        means the decorator has to come in the line **before** `attrs.define`.

    ..  versionadded:: 20.1.0
    ..  versionadded:: 21.1.0 *attribs*

    """
    # Since calling get_type_hints is expensive we cache whether we've
    # done it already.
    if getattr(cls, "__attrs_types_resolved__", None) != cls:
        import typing

        hints = typing.get_type_hints(cls, globalns=globalns, localns=localns)
        for field in fields(cls) if attribs is None else attribs:
            if field.name in hints:
                # Since fields have been frozen we must work around it.
                _obj_setattr(field, "type", hints[field.name])
        # We store the class we resolved so that subclasses know they haven't
        # been resolved.
        cls.__attrs_types_resolved__ = cls

    # Return the class so you can use it as a decorator too.
    return cls
