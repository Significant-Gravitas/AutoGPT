import string
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    TextIO,
    Tuple,
)

from pip._vendor.tomli._re import (
    RE_BIN,
    RE_DATETIME,
    RE_HEX,
    RE_LOCALTIME,
    RE_NUMBER,
    RE_OCT,
    match_to_datetime,
    match_to_localtime,
    match_to_number,
)

if TYPE_CHECKING:
    from re import Pattern


ASCII_CTRL = frozenset(chr(i) for i in range(32)) | frozenset(chr(127))

# Neither of these sets include quotation mark or backslash. They are
# currently handled as separate cases in the parser functions.
ILLEGAL_BASIC_STR_CHARS = ASCII_CTRL - frozenset("\t")
ILLEGAL_MULTILINE_BASIC_STR_CHARS = ASCII_CTRL - frozenset("\t\n\r")

ILLEGAL_LITERAL_STR_CHARS = ILLEGAL_BASIC_STR_CHARS
ILLEGAL_MULTILINE_LITERAL_STR_CHARS = ASCII_CTRL - frozenset("\t\n")

ILLEGAL_COMMENT_CHARS = ILLEGAL_BASIC_STR_CHARS

TOML_WS = frozenset(" \t")
TOML_WS_AND_NEWLINE = TOML_WS | frozenset("\n")
BARE_KEY_CHARS = frozenset(string.ascii_letters + string.digits + "-_")
KEY_INITIAL_CHARS = BARE_KEY_CHARS | frozenset("\"'")

BASIC_STR_ESCAPE_REPLACEMENTS = MappingProxyType(
    {
        "\\b": "\u0008",  # backspace
        "\\t": "\u0009",  # tab
        "\\n": "\u000A",  # linefeed
        "\\f": "\u000C",  # form feed
        "\\r": "\u000D",  # carriage return
        '\\"': "\u0022",  # quote
        "\\\\": "\u005C",  # backslash
    }
)

# Type annotations
ParseFloat = Callable[[str], Any]
Key = Tuple[str, ...]
Pos = int


class TOMLDecodeError(ValueError):
    """An error raised if a document is not valid TOML."""


def load(fp: TextIO, *, parse_float: ParseFloat = float) -> Dict[str, Any]:
    """Parse TOML from a file object."""
    s = fp.read()
    return loads(s, parse_float=parse_float)


def loads(s: str, *, parse_float: ParseFloat = float) -> Dict[str, Any]:  # noqa: C901
    """Parse TOML from a string."""

    # The spec allows converting "\r\n" to "\n", even in string
    # literals. Let's do so to simplify parsing.
    src = s.replace("\r\n", "\n")
    pos = 0
    state = State()

    # Parse one statement at a time
    # (typically means one line in TOML source)
    while True:
        # 1. Skip line leading whitespace
        pos = skip_chars(src, pos, TOML_WS)

        # 2. Parse rules. Expect one of the following:
        #    - end of file
        #    - end of line
        #    - comment
        #    - key/value pair
        #    - append dict to list (and move to its namespace)
        #    - create dict (and move to its namespace)
        # Skip trailing whitespace when applicable.
        try:
            char = src[pos]
        except IndexError:
            break
        if char == "\n":
            pos += 1
            continue
        if char in KEY_INITIAL_CHARS:
            pos = key_value_rule(src, pos, state, parse_float)
            pos = skip_chars(src, pos, TOML_WS)
        elif char == "[":
            try:
                second_char: Optional[str] = src[pos + 1]
            except IndexError:
                second_char = None
            if second_char == "[":
                pos = create_list_rule(src, pos, state)
            else:
                pos = create_dict_rule(src, pos, state)
            pos = skip_chars(src, pos, TOML_WS)
        elif char != "#":
            raise suffixed_err(src, pos, "Invalid statement")

        # 3. Skip comment
        pos = skip_comment(src, pos)

        # 4. Expect end of line or end of file
        try:
            char = src[pos]
        except IndexError:
            break
        if char != "\n":
            raise suffixed_err(
                src, pos, "Expected newline or end of document after a statement"
            )
        pos += 1

    return state.out.dict


class State:
    def __init__(self) -> None:
        # Mutable, read-only
        self.out = NestedDict()
        self.flags = Flags()

        # Immutable, read and write
        self.header_namespace: Key = ()


class Flags:
    """Flags that map to parsed keys/namespaces."""

    # Marks an immutable namespace (inline array or inline table).
    FROZEN = 0
    # Marks a nest that has been explicitly created and can no longer
    # be opened using the "[table]" syntax.
    EXPLICIT_NEST = 1

    def __init__(self) -> None:
        self._flags: Dict[str, dict] = {}

    def unset_all(self, key: Key) -> None:
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return
            cont = cont[k]["nested"]
        cont.pop(key[-1], None)

    def set_for_relative_key(self, head_key: Key, rel_key: Key, flag: int) -> None:
        cont = self._flags
        for k in head_key:
            if k not in cont:
                cont[k] = {"flags": set(), "recursive_flags": set(), "nested": {}}
            cont = cont[k]["nested"]
        for k in rel_key:
            if k in cont:
                cont[k]["flags"].add(flag)
            else:
                cont[k] = {"flags": {flag}, "recursive_flags": set(), "nested": {}}
            cont = cont[k]["nested"]

    def set(self, key: Key, flag: int, *, recursive: bool) -> None:  # noqa: A003
        cont = self._flags
        key_parent, key_stem = key[:-1], key[-1]
        for k in key_parent:
            if k not in cont:
                cont[k] = {"flags": set(), "recursive_flags": set(), "nested": {}}
            cont = cont[k]["nested"]
        if key_stem not in cont:
            cont[key_stem] = {"flags": set(), "recursive_flags": set(), "nested": {}}
        cont[key_stem]["recursive_flags" if recursive else "flags"].add(flag)

    def is_(self, key: Key, flag: int) -> bool:
        if not key:
            return False  # document root has no flags
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return False
            inner_cont = cont[k]
            if flag in inner_cont["recursive_flags"]:
                return True
            cont = inner_cont["nested"]
        key_stem = key[-1]
        if key_stem in cont:
            cont = cont[key_stem]
            return flag in cont["flags"] or flag in cont["recursive_flags"]
        return False


class NestedDict:
    def __init__(self) -> None:
        # The parsed content of the TOML document
        self.dict: Dict[str, Any] = {}

    def get_or_create_nest(
        self,
        key: Key,
        *,
        access_lists: bool = True,
    ) -> dict:
        cont: Any = self.dict
        for k in key:
            if k not in cont:
                cont[k] = {}
            cont = cont[k]
            if access_lists and isinstance(cont, list):
                cont = cont[-1]
            if not isinstance(cont, dict):
                raise KeyError("There is no nest behind this key")
        return cont

    def append_nest_to_list(self, key: Key) -> None:
        cont = self.get_or_create_nest(key[:-1])
        last_key = key[-1]
        if last_key in cont:
            list_ = cont[last_key]
            if not isinstance(list_, list):
                raise KeyError("An object other than list found behind this key")
            list_.append({})
        else:
            cont[last_key] = [{}]


def skip_chars(src: str, pos: Pos, chars: Iterable[str]) -> Pos:
    try:
        while src[pos] in chars:
            pos += 1
    except IndexError:
        pass
    return pos


def skip_until(
    src: str,
    pos: Pos,
    expect: str,
    *,
    error_on: FrozenSet[str],
    error_on_eof: bool,
) -> Pos:
    try:
        new_pos = src.index(expect, pos)
    except ValueError:
        new_pos = len(src)
        if error_on_eof:
            raise suffixed_err(src, new_pos, f'Expected "{expect!r}"')

    bad_chars = error_on.intersection(src[pos:new_pos])
    if bad_chars:
        bad_char = next(iter(bad_chars))
        bad_pos = src.index(bad_char, pos)
        raise suffixed_err(src, bad_pos, f'Found invalid character "{bad_char!r}"')
    return new_pos


def skip_comment(src: str, pos: Pos) -> Pos:
    try:
        char: Optional[str] = src[pos]
    except IndexError:
        char = None
    if char == "#":
        return skip_until(
            src, pos + 1, "\n", error_on=ILLEGAL_COMMENT_CHARS, error_on_eof=False
        )
    return pos


def skip_comments_and_array_ws(src: str, pos: Pos) -> Pos:
    while True:
        pos_before_skip = pos
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        pos = skip_comment(src, pos)
        if pos == pos_before_skip:
            return pos


def create_dict_rule(src: str, pos: Pos, state: State) -> Pos:
    pos += 1  # Skip "["
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)

    if state.flags.is_(key, Flags.EXPLICIT_NEST) or state.flags.is_(key, Flags.FROZEN):
        raise suffixed_err(src, pos, f"Can not declare {key} twice")
    state.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        state.out.get_or_create_nest(key)
    except KeyError:
        raise suffixed_err(src, pos, "Can not overwrite a value")
    state.header_namespace = key

    if src[pos : pos + 1] != "]":
        raise suffixed_err(src, pos, 'Expected "]" at the end of a table declaration')
    return pos + 1


def create_list_rule(src: str, pos: Pos, state: State) -> Pos:
    pos += 2  # Skip "[["
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)

    if state.flags.is_(key, Flags.FROZEN):
        raise suffixed_err(src, pos, f"Can not mutate immutable namespace {key}")
    # Free the namespace now that it points to another empty list item...
    state.flags.unset_all(key)
    # ...but this key precisely is still prohibited from table declaration
    state.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        state.out.append_nest_to_list(key)
    except KeyError:
        raise suffixed_err(src, pos, "Can not overwrite a value")
    state.header_namespace = key

    end_marker = src[pos : pos + 2]
    if end_marker != "]]":
        raise suffixed_err(
            src,
            pos,
            f'Found "{end_marker!r}" at the end of an array declaration.'
            ' Expected "]]"',
        )
    return pos + 2


def key_value_rule(src: str, pos: Pos, state: State, parse_float: ParseFloat) -> Pos:
    pos, key, value = parse_key_value_pair(src, pos, parse_float)
    key_parent, key_stem = key[:-1], key[-1]
    abs_key_parent = state.header_namespace + key_parent

    if state.flags.is_(abs_key_parent, Flags.FROZEN):
        raise suffixed_err(
            src, pos, f"Can not mutate immutable namespace {abs_key_parent}"
        )
    # Containers in the relative path can't be opened with the table syntax after this
    state.flags.set_for_relative_key(state.header_namespace, key, Flags.EXPLICIT_NEST)
    try:
        nest = state.out.get_or_create_nest(abs_key_parent)
    except KeyError:
        raise suffixed_err(src, pos, "Can not overwrite a value")
    if key_stem in nest:
        raise suffixed_err(src, pos, "Can not overwrite a value")
    # Mark inline table and array namespaces recursively immutable
    if isinstance(value, (dict, list)):
        abs_key = state.header_namespace + key
        state.flags.set(abs_key, Flags.FROZEN, recursive=True)
    nest[key_stem] = value
    return pos


def parse_key_value_pair(
    src: str, pos: Pos, parse_float: ParseFloat
) -> Tuple[Pos, Key, Any]:
    pos, key = parse_key(src, pos)
    try:
        char: Optional[str] = src[pos]
    except IndexError:
        char = None
    if char != "=":
        raise suffixed_err(src, pos, 'Expected "=" after a key in a key/value pair')
    pos += 1
    pos = skip_chars(src, pos, TOML_WS)
    pos, value = parse_value(src, pos, parse_float)
    return pos, key, value


def parse_key(src: str, pos: Pos) -> Tuple[Pos, Key]:
    pos, key_part = parse_key_part(src, pos)
    key = [key_part]
    pos = skip_chars(src, pos, TOML_WS)
    while True:
        try:
            char: Optional[str] = src[pos]
        except IndexError:
            char = None
        if char != ".":
            return pos, tuple(key)
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)
        pos, key_part = parse_key_part(src, pos)
        key.append(key_part)
        pos = skip_chars(src, pos, TOML_WS)


def parse_key_part(src: str, pos: Pos) -> Tuple[Pos, str]:
    try:
        char: Optional[str] = src[pos]
    except IndexError:
        char = None
    if char in BARE_KEY_CHARS:
        start_pos = pos
        pos = skip_chars(src, pos, BARE_KEY_CHARS)
        return pos, src[start_pos:pos]
    if char == "'":
        return parse_literal_str(src, pos)
    if char == '"':
        return parse_one_line_basic_str(src, pos)
    raise suffixed_err(src, pos, "Invalid initial character for a key part")


def parse_one_line_basic_str(src: str, pos: Pos) -> Tuple[Pos, str]:
    pos += 1
    return parse_basic_str(src, pos, multiline=False)


def parse_array(src: str, pos: Pos, parse_float: ParseFloat) -> Tuple[Pos, list]:
    pos += 1
    array: list = []

    pos = skip_comments_and_array_ws(src, pos)
    if src[pos : pos + 1] == "]":
        return pos + 1, array
    while True:
        pos, val = parse_value(src, pos, parse_float)
        array.append(val)
        pos = skip_comments_and_array_ws(src, pos)

        c = src[pos : pos + 1]
        if c == "]":
            return pos + 1, array
        if c != ",":
            raise suffixed_err(src, pos, "Unclosed array")
        pos += 1

        pos = skip_comments_and_array_ws(src, pos)
        if src[pos : pos + 1] == "]":
            return pos + 1, array


def parse_inline_table(src: str, pos: Pos, parse_float: ParseFloat) -> Tuple[Pos, dict]:
    pos += 1
    nested_dict = NestedDict()
    flags = Flags()

    pos = skip_chars(src, pos, TOML_WS)
    if src[pos : pos + 1] == "}":
        return pos + 1, nested_dict.dict
    while True:
        pos, key, value = parse_key_value_pair(src, pos, parse_float)
        key_parent, key_stem = key[:-1], key[-1]
        if flags.is_(key, Flags.FROZEN):
            raise suffixed_err(src, pos, f"Can not mutate immutable namespace {key}")
        try:
            nest = nested_dict.get_or_create_nest(key_parent, access_lists=False)
        except KeyError:
            raise suffixed_err(src, pos, "Can not overwrite a value")
        if key_stem in nest:
            raise suffixed_err(src, pos, f'Duplicate inline table key "{key_stem}"')
        nest[key_stem] = value
        pos = skip_chars(src, pos, TOML_WS)
        c = src[pos : pos + 1]
        if c == "}":
            return pos + 1, nested_dict.dict
        if c != ",":
            raise suffixed_err(src, pos, "Unclosed inline table")
        if isinstance(value, (dict, list)):
            flags.set(key, Flags.FROZEN, recursive=True)
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)


def parse_basic_str_escape(
    src: str, pos: Pos, *, multiline: bool = False
) -> Tuple[Pos, str]:
    escape_id = src[pos : pos + 2]
    pos += 2
    if multiline and escape_id in {"\\ ", "\\\t", "\\\n"}:
        # Skip whitespace until next non-whitespace character or end of
        # the doc. Error if non-whitespace is found before newline.
        if escape_id != "\\\n":
            pos = skip_chars(src, pos, TOML_WS)
            char = src[pos : pos + 1]
            if not char:
                return pos, ""
            if char != "\n":
                raise suffixed_err(src, pos, 'Unescaped "\\" in a string')
            pos += 1
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        return pos, ""
    if escape_id == "\\u":
        return parse_hex_char(src, pos, 4)
    if escape_id == "\\U":
        return parse_hex_char(src, pos, 8)
    try:
        return pos, BASIC_STR_ESCAPE_REPLACEMENTS[escape_id]
    except KeyError:
        if len(escape_id) != 2:
            raise suffixed_err(src, pos, "Unterminated string")
        raise suffixed_err(src, pos, 'Unescaped "\\" in a string')


def parse_basic_str_escape_multiline(src: str, pos: Pos) -> Tuple[Pos, str]:
    return parse_basic_str_escape(src, pos, multiline=True)


def parse_hex_char(src: str, pos: Pos, hex_len: int) -> Tuple[Pos, str]:
    hex_str = src[pos : pos + hex_len]
    if len(hex_str) != hex_len or any(c not in string.hexdigits for c in hex_str):
        raise suffixed_err(src, pos, "Invalid hex value")
    pos += hex_len
    hex_int = int(hex_str, 16)
    if not is_unicode_scalar_value(hex_int):
        raise suffixed_err(src, pos, "Escaped character is not a Unicode scalar value")
    return pos, chr(hex_int)


def parse_literal_str(src: str, pos: Pos) -> Tuple[Pos, str]:
    pos += 1  # Skip starting apostrophe
    start_pos = pos
    pos = skip_until(
        src, pos, "'", error_on=ILLEGAL_LITERAL_STR_CHARS, error_on_eof=True
    )
    return pos + 1, src[start_pos:pos]  # Skip ending apostrophe


def parse_multiline_str(src: str, pos: Pos, *, literal: bool) -> Tuple[Pos, str]:
    pos += 3
    if src[pos : pos + 1] == "\n":
        pos += 1

    if literal:
        delim = "'"
        end_pos = skip_until(
            src,
            pos,
            "'''",
            error_on=ILLEGAL_MULTILINE_LITERAL_STR_CHARS,
            error_on_eof=True,
        )
        result = src[pos:end_pos]
        pos = end_pos + 3
    else:
        delim = '"'
        pos, result = parse_basic_str(src, pos, multiline=True)

    # Add at maximum two extra apostrophes/quotes if the end sequence
    # is 4 or 5 chars long instead of just 3.
    if src[pos : pos + 1] != delim:
        return pos, result
    pos += 1
    if src[pos : pos + 1] != delim:
        return pos, result + delim
    pos += 1
    return pos, result + (delim * 2)


def parse_basic_str(src: str, pos: Pos, *, multiline: bool) -> Tuple[Pos, str]:
    if multiline:
        error_on = ILLEGAL_MULTILINE_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape_multiline
    else:
        error_on = ILLEGAL_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape
    result = ""
    start_pos = pos
    while True:
        try:
            char = src[pos]
        except IndexError:
            raise suffixed_err(src, pos, "Unterminated string")
        if char == '"':
            if not multiline:
                return pos + 1, result + src[start_pos:pos]
            if src[pos + 1 : pos + 3] == '""':
                return pos + 3, result + src[start_pos:pos]
            pos += 1
            continue
        if char == "\\":
            result += src[start_pos:pos]
            pos, parsed_escape = parse_escapes(src, pos)
            result += parsed_escape
            start_pos = pos
            continue
        if char in error_on:
            raise suffixed_err(src, pos, f'Illegal character "{char!r}"')
        pos += 1


def parse_regex(src: str, pos: Pos, regex: "Pattern") -> Tuple[Pos, str]:
    match = regex.match(src, pos)
    if not match:
        raise suffixed_err(src, pos, "Unexpected sequence")
    return match.end(), match.group()


def parse_value(  # noqa: C901
    src: str, pos: Pos, parse_float: ParseFloat
) -> Tuple[Pos, Any]:
    try:
        char: Optional[str] = src[pos]
    except IndexError:
        char = None

    # Basic strings
    if char == '"':
        if src[pos + 1 : pos + 3] == '""':
            return parse_multiline_str(src, pos, literal=False)
        return parse_one_line_basic_str(src, pos)

    # Literal strings
    if char == "'":
        if src[pos + 1 : pos + 3] == "''":
            return parse_multiline_str(src, pos, literal=True)
        return parse_literal_str(src, pos)

    # Booleans
    if char == "t":
        if src[pos + 1 : pos + 4] == "rue":
            return pos + 4, True
    if char == "f":
        if src[pos + 1 : pos + 5] == "alse":
            return pos + 5, False

    # Dates and times
    datetime_match = RE_DATETIME.match(src, pos)
    if datetime_match:
        try:
            datetime_obj = match_to_datetime(datetime_match)
        except ValueError:
            raise suffixed_err(src, pos, "Invalid date or datetime")
        return datetime_match.end(), datetime_obj
    localtime_match = RE_LOCALTIME.match(src, pos)
    if localtime_match:
        return localtime_match.end(), match_to_localtime(localtime_match)

    # Non-decimal integers
    if char == "0":
        second_char = src[pos + 1 : pos + 2]
        if second_char == "x":
            pos, hex_str = parse_regex(src, pos + 2, RE_HEX)
            return pos, int(hex_str, 16)
        if second_char == "o":
            pos, oct_str = parse_regex(src, pos + 2, RE_OCT)
            return pos, int(oct_str, 8)
        if second_char == "b":
            pos, bin_str = parse_regex(src, pos + 2, RE_BIN)
            return pos, int(bin_str, 2)

    # Decimal integers and "normal" floats.
    # The regex will greedily match any type starting with a decimal
    # char, so needs to be located after handling of non-decimal ints,
    # and dates and times.
    number_match = RE_NUMBER.match(src, pos)
    if number_match:
        return number_match.end(), match_to_number(number_match, parse_float)

    # Arrays
    if char == "[":
        return parse_array(src, pos, parse_float)

    # Inline tables
    if char == "{":
        return parse_inline_table(src, pos, parse_float)

    # Special floats
    first_three = src[pos : pos + 3]
    if first_three in {"inf", "nan"}:
        return pos + 3, parse_float(first_three)
    first_four = src[pos : pos + 4]
    if first_four in {"-inf", "+inf", "-nan", "+nan"}:
        return pos + 4, parse_float(first_four)

    raise suffixed_err(src, pos, "Invalid value")


def suffixed_err(src: str, pos: Pos, msg: str) -> TOMLDecodeError:
    """Return a `TOMLDecodeError` where error message is suffixed with
    coordinates in source."""

    def coord_repr(src: str, pos: Pos) -> str:
        if pos >= len(src):
            return "end of document"
        line = src.count("\n", 0, pos) + 1
        if line == 1:
            column = pos + 1
        else:
            column = pos - src.rindex("\n", 0, pos)
        return f"line {line}, column {column}"

    return TOMLDecodeError(f"{msg} (at {coord_repr(src, pos)})")


def is_unicode_scalar_value(codepoint: int) -> bool:
    return (0 <= codepoint <= 55295) or (57344 <= codepoint <= 1114111)
