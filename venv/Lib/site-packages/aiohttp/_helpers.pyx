cdef class reify:
    """Use as a class method decorator.  It operates almost exactly like
    the Python `@property` decorator, but it puts the result of the
    method it decorates into the instance dict after the first call,
    effectively replacing the function it decorates with an instance
    variable.  It is, in Python parlance, a data descriptor.

    """

    cdef object wrapped
    cdef object name

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.name = wrapped.__name__

    @property
    def __doc__(self):
        return self.wrapped.__doc__

    def __get__(self, inst, owner):
        try:
            try:
                return inst._cache[self.name]
            except KeyError:
                val = self.wrapped(inst)
                inst._cache[self.name] = val
                return val
        except AttributeError:
            if inst is None:
                return self
            raise

    def __set__(self, inst, value):
        raise AttributeError("reified property is read-only")
