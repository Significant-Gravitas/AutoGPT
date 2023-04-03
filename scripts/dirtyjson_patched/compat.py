"""Python 3 compatibility shims
"""
import sys
if sys.version_info[0] < 3:
    PY2 = True

    def b(s):
        return s

    def u(s):
        # noinspection PyUnresolvedReferences
        return unicode(s, 'unicode_escape')

    # noinspection PyUnresolvedReferences
    import cStringIO as StringIO
    # noinspection PyUnresolvedReferences
    StringIO = BytesIO = StringIO.StringIO
    # noinspection PyUnresolvedReferences
    text_type = unicode
    binary_type = str
    # noinspection PyUnresolvedReferences
    string_types = (basestring,)
    # noinspection PyUnresolvedReferences
    integer_types = (int, long)
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    unichr = unichr

    def fromhex(s):
        return s.decode('hex')

else:
    PY2 = False
    import codecs

    def b(s):
        return codecs.latin_1_encode(s)[0]

    def u(s):
        return s

    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    text_type = str
    binary_type = bytes
    string_types = (str,)
    integer_types = (int,)

    def unichr(s):
        return u(chr(s))

    def fromhex(s):
        return bytes.fromhex(s)

long_type = integer_types[-1]
