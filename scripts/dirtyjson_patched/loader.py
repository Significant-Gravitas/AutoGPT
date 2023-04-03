"""Implementation of JSONDecoder
"""
from __future__ import absolute_import
import re
import sys
import struct
from .compat import fromhex, u, text_type, binary_type, PY2, unichr
from dirtyjson.attributed_containers import AttributedDict, AttributedList
from .error import Error


def _floatconstants():
    _BYTES = fromhex('7FF80000000000007FF0000000000000')
    # The struct module in Python 2.4 would get frexp() out of range here
    # when an endian is specified in the format string. Fixed in Python 2.5+
    if sys.byteorder != 'big':
        _BYTES = _BYTES[:8][::-1] + _BYTES[8:][::-1]
    nan, inf = struct.unpack('dd', _BYTES)
    return nan, inf, -inf

NaN, PosInf, NegInf = _floatconstants()

_CONSTANTS = {
    'null': None,
    'true': True,
    'false': False,
    '-Infinity': NegInf,
    'Infinity': PosInf,
    'NaN': NaN,
}

CONSTANT_RE = re.compile('(%s)' % '|'.join(_CONSTANTS))
NUMBER_RE = re.compile(r'(-?(?:(?:0x)?\d+))(\.\d+)?([eE][-+]?\d+)?')
STRINGCHUNK_DOUBLEQUOTE = re.compile(r'(.*?)(["\\\x00-\x1f])')
STRINGCHUNK_SINGLEQUOTE = re.compile(r"(.*?)(['\\\x00-\x1f])")
UNQUOTED_KEYNAME = re.compile(r"([\w_]+[\w\d_]*)")
WHITESPACE_STR = ' \t\n\r'
WHITESPACE = re.compile('[%s]*' % WHITESPACE_STR, re.VERBOSE | re.MULTILINE | re.DOTALL)

BACKSLASH = {
    '"': u('"'), '\\': u('\u005c'), '/': u('/'),
    'b': u('\b'), 'f': u('\f'), 'n': u('\n'), 'r': u('\r'), 't': u('\t'),
}
DEFAULT_ENCODING = "utf-8"


class DirtyJSONLoader(object):
    """JSON decoder that can handle muck in the file

    Performs the following translations in decoding by default:

    +---------------+-------------------+
    | JSON          | Python            |
    +===============+===================+
    | object        | AttributedDict    |
    +---------------+-------------------+
    | array         | list              |
    +---------------+-------------------+
    | string        | unicode           |
    +---------------+-------------------+
    | number (int)  | int, long         |
    +---------------+-------------------+
    | number (real) | float             |
    +---------------+-------------------+
    | true          | True              |
    +---------------+-------------------+
    | false         | False             |
    +---------------+-------------------+
    | null          | None              |
    +---------------+-------------------+

    It also understands ``NaN``, ``Infinity``, and ``-Infinity`` as
    their corresponding ``float`` values, which is outside the JSON spec.

    """

    def __init__(self, content, encoding=None, parse_float=None, parse_int=None,
                 parse_constant=None):
        self.encoding = encoding or DEFAULT_ENCODING
        self.parse_float = parse_float or float
        self.parse_int = parse_int or int
        self.parse_constant = parse_constant or _CONSTANTS.__getitem__
        self.memo = {}
        if not PY2 and isinstance(content, binary_type):
            self.content = content.decode(self.encoding)
        else:
            self.content = content
        self.end = len(self.content)
        self.lineno = 1
        self.current_line_pos = 0
        self.pos = 0
        self.expecting = 'Expecting value'

    def _next_character(self):
        try:
            nextchar = self.content[self.pos]
            self.pos += 1
            return nextchar
        except IndexError:
            raise Error(self.expecting, self.content, self.pos)

    def _next_character_after_whitespace(self):
        try:
            nextchar = self.content[self.pos]
            if nextchar in WHITESPACE_STR:
                self._skip_whitespace()
                nextchar = self.content[self.pos]
            self.pos += 1
            return nextchar
        except IndexError:
            return ''

    def _skip_whitespace(self):
        while True:
            self._skip_forward_to(WHITESPACE.match(self.content, self.pos).end())
            if self.pos > self.end - 2:
                break
            two_chars = self.content[self.pos:self.pos + 2]
            if two_chars == '//' or two_chars == '/*':
                terminator = '\n' if two_chars == '//' else '*/'
                lf = self.content.index(terminator, self.pos)
                if lf >= 0:
                    self._skip_forward_to(lf + len(terminator))
                else:
                    self._skip_forward_to(self.end)
                    break
            else:
                break

    def _skip_forward_to(self, end):
        if end != self.pos:
            linefeeds = self.content.count('\n', self.pos, end)
            if linefeeds:
                self.lineno += linefeeds
                rpos = self.content.rfind('\n', self.pos, end)
                self.current_line_pos = rpos + 1
            self.pos = end

    def _current_position(self, offset=0):
        return self.lineno, self.pos - self.current_line_pos + 1 + offset

    def scan(self):
        self.expecting = 'Expecting value'
        nextchar = self._next_character()

        if nextchar == '"' or nextchar == "'":
            return self.parse_string(nextchar)
        if nextchar == '{':
            return self.parse_object()
        if nextchar == '[':
            return self.parse_array()

        self.pos -= 1
        m = CONSTANT_RE.match(self.content, self.pos)
        if m:
            self.pos = m.end()
            return self.parse_constant(m.groups()[0])

        m = NUMBER_RE.match(self.content, self.pos)
        if m:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = self.parse_float(integer + (frac or '') + (exp or ''))
            else:
                try:
                    res = self.parse_int(int(integer, 0))
                except ValueError:
                    if integer[0] == '0':
                        integer = '0o' + integer[1:]
                        res = self.parse_int(int(integer, 0))
                    else:
                        raise
            self.pos = m.end()
            return res
        raise Error(self.expecting, self.content, self.pos)

    def parse_string(self, terminating_character,
                     _b=BACKSLASH, _join=u('').join,
                     _py2=PY2, _maxunicode=sys.maxunicode):
        """Scan the string for a JSON string. End is the index of the
        character in string after the quote that started the JSON string.
        Unescapes all valid JSON string escape sequences and raises ValueError
        on attempt to decode an invalid string.

        Returns a tuple of the decoded string and the index of the character in
        string after the end quote."""
        _m = STRINGCHUNK_DOUBLEQUOTE.match if terminating_character == '"' else STRINGCHUNK_SINGLEQUOTE.match
        chunks = []
        _append = chunks.append
        begin = self.pos - 1
        while 1:
            chunk = _m(self.content, self.pos)
            if chunk is None:
                raise Error(
                    "Unterminated string starting at", self.content, begin)
            self.pos = chunk.end()
            content, terminator = chunk.groups()
            # Content is contains zero or more unescaped string characters
            if content:
                if _py2 and not isinstance(content, text_type):
                    content = text_type(content, self.encoding)
                _append(content)
            # Terminator is the end of string, a literal control character,
            # or a backslash denoting that an escape sequence follows
            if terminator == terminating_character:
                break
            elif terminator != '\\':
                _append(terminator)
                continue
            try:
                esc = self.content[self.pos]
            except IndexError:
                raise Error(
                    "Unterminated string starting at", self.content, begin)
            # If not a unicode escape sequence, must be in the lookup table
            if esc != 'u':
                try:
                    char = _b[esc]
                except KeyError:
                    msg = "Invalid \\X escape sequence %r"
                    raise Error(msg, self.content, self.pos)
                self.pos += 1
            else:
                # Unicode escape sequence
                msg = "Invalid \\uXXXX escape sequence"
                esc = self.content[self.pos + 1:self.pos + 5]
                esc_x = esc[1:2]
                if len(esc) != 4 or esc_x == 'x' or esc_x == 'X':
                    raise Error(msg, self.content, self.pos - 1)
                try:
                    uni = int(esc, 16)
                except ValueError:
                    raise Error(msg, self.content, self.pos - 1)
                self.pos += 5
                # Check for surrogate pair on UCS-4 systems
                # Note that this will join high/low surrogate pairs
                # but will also pass unpaired surrogates through
                if _maxunicode > 65535 and uni & 0xfc00 == 0xd800 and self.content[self.pos:self.pos + 2] == '\\u':
                    esc2 = self.content[self.pos + 2:self.pos + 6]
                    esc_x = esc2[1:2]
                    if len(esc2) == 4 and not (esc_x == 'x' or esc_x == 'X'):
                        try:
                            uni2 = int(esc2, 16)
                        except ValueError:
                            raise Error(msg, self.content, self.pos)
                        if uni2 & 0xfc00 == 0xdc00:
                            uni = 0x10000 + (((uni - 0xd800) << 10) |
                                             (uni2 - 0xdc00))
                            self.pos += 6
                char = unichr(uni)
            # Append the unescaped character
            _append(char)
        return _join(chunks)

    def parse_object(self):
        # Backwards compatibility
        memo_get = self.memo.setdefault
        obj = AttributedDict()
        # Use a slice to prevent IndexError from being raised, the following
        # check will raise a more specific ValueError if the string is empty
        nextchar = self._next_character_after_whitespace()
        # Trivial empty object
        while True:
            if nextchar == '}':
                break
            key_pos = self._current_position(len(nextchar))
            if nextchar == '"' or nextchar == "'":
                key = self.parse_string(nextchar)
            else:
                chunk = UNQUOTED_KEYNAME.match(self.content, self.pos - 1)
                if chunk is None:
                    raise Error(
                        "Expecting property name",
                        self.content, self.pos)
                self.pos = chunk.end()
                key = chunk.groups()[0]
            key = memo_get(key, key)

            # To skip some function call overhead we optimize the fast paths where
            # the JSON key separator is ": " or just ":".
            if self._next_character_after_whitespace() != ':':
                raise Error("Expecting ':' delimiter", self.content, self.pos)

            self._skip_whitespace()
            value_pos = self._current_position()
            value = self.scan()
            obj.add_with_attributes(key, value, {'key': key_pos, 'value': value_pos})

            nextchar = self._next_character_after_whitespace()
            if nextchar == '}':
                break
            elif nextchar != ',':
                raise Error("Expecting ',' delimiter or '}'", self.content, self.pos - len(nextchar))

            nextchar = self._next_character_after_whitespace()

        return obj

    def parse_array(self):
        values = AttributedList()
        nextchar = self._next_character_after_whitespace()
        # Look-ahead for trivial empty array
        if nextchar == ']':
            return values
        elif nextchar == '':
            raise Error("Expecting value or ']'", self.content, self.pos)
        _append = values.append
        while True:
            if nextchar == ']':
                break
            self.pos -= len(nextchar)
            value_pos = self._current_position()
            value = self.scan()
            _append(value, value_pos)
            nextchar = self._next_character_after_whitespace()
            if nextchar == ']':
                break
            elif nextchar != ',':
                raise Error("Expecting ',' delimiter or ']'", self.content, self.pos - len(nextchar))

            nextchar = self._next_character_after_whitespace()

        return values

    def decode(self, search_for_first_object=False, start_index=0):
        """Return the Python representation of ``s`` (a ``str`` or ``unicode``
        instance containing a JSON document)
        """
        if start_index:
            self._skip_forward_to(start_index)

        if search_for_first_object:
            i = self.content.find('[', self.pos)
            o = self.content.find('{', self.pos)
            if i > o >= self.pos:
                i = o
            if i >= self.pos:
                self._skip_forward_to(i)

        self._skip_whitespace()
        obj = self.scan()
        return obj
