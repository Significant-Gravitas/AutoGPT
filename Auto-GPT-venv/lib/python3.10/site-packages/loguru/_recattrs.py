import pickle
from collections import namedtuple


class RecordLevel:
    __slots__ = ("name", "no", "icon")

    def __init__(self, name, no, icon):
        self.name = name
        self.no = no
        self.icon = icon

    def __repr__(self):
        return "(name=%r, no=%r, icon=%r)" % (self.name, self.no, self.icon)

    def __format__(self, spec):
        return self.name.__format__(spec)


class RecordFile:
    __slots__ = ("name", "path")

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __repr__(self):
        return "(name=%r, path=%r)" % (self.name, self.path)

    def __format__(self, spec):
        return self.name.__format__(spec)


class RecordThread:
    __slots__ = ("id", "name")

    def __init__(self, id_, name):
        self.id = id_
        self.name = name

    def __repr__(self):
        return "(id=%r, name=%r)" % (self.id, self.name)

    def __format__(self, spec):
        return self.id.__format__(spec)


class RecordProcess:
    __slots__ = ("id", "name")

    def __init__(self, id_, name):
        self.id = id_
        self.name = name

    def __repr__(self):
        return "(id=%r, name=%r)" % (self.id, self.name)

    def __format__(self, spec):
        return self.id.__format__(spec)


class RecordException(namedtuple("RecordException", ("type", "value", "traceback"))):
    def __repr__(self):
        return "(type=%r, value=%r, traceback=%r)" % (self.type, self.value, self.traceback)

    def __reduce__(self):
        # The traceback is not picklable so we need to remove it. Also, some custom exception
        # values aren't picklable either. For user convenience, we try first to serialize it and
        # we remove the value in case or error. As an optimization, we could have re-used the
        # dumped value during unpickling, but this requires using "pickle.loads()" which is
        # flagged as insecure by some security tools.
        try:
            pickle.dumps(self.value)
        except pickle.PickleError:
            return (RecordException, (self.type, None, None))
        else:
            return (RecordException, (self.type, self.value, None))
