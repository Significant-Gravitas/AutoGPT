# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# This implementation of the immutable decorator requires python >=
# 3.7, and is significantly more storage efficient when making classes
# with slots immutable.  It's also faster.

import contextvars
import inspect


_in__init__ = contextvars.ContextVar("_immutable_in__init__", default=False)


class _Immutable:
    """Immutable mixin class"""

    # We set slots to the empty list to say "we don't have any attributes".
    # We do this so that if we're mixed in with a class with __slots__, we
    # don't cause a __dict__ to be added which would waste space.

    __slots__ = ()

    def __setattr__(self, name, value):
        if _in__init__.get() is not self:
            raise TypeError("object doesn't support attribute assignment")
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if _in__init__.get() is not self:
            raise TypeError("object doesn't support attribute assignment")
        else:
            super().__delattr__(name)


def _immutable_init(f):
    def nf(*args, **kwargs):
        previous = _in__init__.set(args[0])
        try:
            # call the actual __init__
            f(*args, **kwargs)
        finally:
            _in__init__.reset(previous)

    nf.__signature__ = inspect.signature(f)
    return nf


def immutable(cls):
    if _Immutable in cls.__mro__:
        # Some ancestor already has the mixin, so just make sure we keep
        # following the __init__ protocol.
        cls.__init__ = _immutable_init(cls.__init__)
        if hasattr(cls, "__setstate__"):
            cls.__setstate__ = _immutable_init(cls.__setstate__)
        ncls = cls
    else:
        # Mixin the Immutable class and follow the __init__ protocol.
        class ncls(_Immutable, cls):
            # We have to do the __slots__ declaration here too!
            __slots__ = ()

            @_immutable_init
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            if hasattr(cls, "__setstate__"):

                @_immutable_init
                def __setstate__(self, *args, **kwargs):
                    super().__setstate__(*args, **kwargs)

        # make ncls have the same name and module as cls
        ncls.__name__ = cls.__name__
        ncls.__qualname__ = cls.__qualname__
        ncls.__module__ = cls.__module__
    return ncls
