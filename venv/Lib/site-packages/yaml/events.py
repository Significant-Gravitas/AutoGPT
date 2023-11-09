
# Abstract classes.

class Event(object):
    def __init__(self, start_mark=None, end_mark=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
    def __repr__(self):
        attributes = [key for key in ['anchor', 'tag', 'implicit', 'value']
                if hasattr(self, key)]
        arguments = ', '.join(['%s=%r' % (key, getattr(self, key))
                for key in attributes])
        return '%s(%s)' % (self.__class__.__name__, arguments)

class NodeEvent(Event):
    def __init__(self, anchor, start_mark=None, end_mark=None):
        self.anchor = anchor
        self.start_mark = start_mark
        self.end_mark = end_mark

class CollectionStartEvent(NodeEvent):
    def __init__(self, anchor, tag, implicit, start_mark=None, end_mark=None,
            flow_style=None):
        self.anchor = anchor
        self.tag = tag
        self.implicit = implicit
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.flow_style = flow_style

class CollectionEndEvent(Event):
    pass

# Implementations.

class StreamStartEvent(Event):
    def __init__(self, start_mark=None, end_mark=None, encoding=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.encoding = encoding

class StreamEndEvent(Event):
    pass

class DocumentStartEvent(Event):
    def __init__(self, start_mark=None, end_mark=None,
            explicit=None, version=None, tags=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.explicit = explicit
        self.version = version
        self.tags = tags

class DocumentEndEvent(Event):
    def __init__(self, start_mark=None, end_mark=None,
            explicit=None):
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.explicit = explicit

class AliasEvent(NodeEvent):
    pass

class ScalarEvent(NodeEvent):
    def __init__(self, anchor, tag, implicit, value,
            start_mark=None, end_mark=None, style=None):
        self.anchor = anchor
        self.tag = tag
        self.implicit = implicit
        self.value = value
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.style = style

class SequenceStartEvent(CollectionStartEvent):
    pass

class SequenceEndEvent(CollectionEndEvent):
    pass

class MappingStartEvent(CollectionStartEvent):
    pass

class MappingEndEvent(CollectionEndEvent):
    pass

