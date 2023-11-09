
__all__ = [
    'CBaseLoader', 'CSafeLoader', 'CFullLoader', 'CUnsafeLoader', 'CLoader',
    'CBaseDumper', 'CSafeDumper', 'CDumper'
]

from yaml._yaml import CParser, CEmitter

from .constructor import *

from .serializer import *
from .representer import *

from .resolver import *

class CBaseLoader(CParser, BaseConstructor, BaseResolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        BaseConstructor.__init__(self)
        BaseResolver.__init__(self)

class CSafeLoader(CParser, SafeConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        SafeConstructor.__init__(self)
        Resolver.__init__(self)

class CFullLoader(CParser, FullConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        FullConstructor.__init__(self)
        Resolver.__init__(self)

class CUnsafeLoader(CParser, UnsafeConstructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        UnsafeConstructor.__init__(self)
        Resolver.__init__(self)

class CLoader(CParser, Constructor, Resolver):

    def __init__(self, stream):
        CParser.__init__(self, stream)
        Constructor.__init__(self)
        Resolver.__init__(self)

class CBaseDumper(CEmitter, BaseRepresenter, BaseResolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        CEmitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width, encoding=encoding,
                allow_unicode=allow_unicode, line_break=line_break,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        Representer.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

class CSafeDumper(CEmitter, SafeRepresenter, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        CEmitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width, encoding=encoding,
                allow_unicode=allow_unicode, line_break=line_break,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        SafeRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

class CDumper(CEmitter, Serializer, Representer, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        CEmitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width, encoding=encoding,
                allow_unicode=allow_unicode, line_break=line_break,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        Representer.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)

