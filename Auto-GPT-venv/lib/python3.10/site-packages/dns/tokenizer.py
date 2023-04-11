# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2017 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Tokenize DNS zone file format"""

from typing import Any, Optional, List, Tuple

import io
import sys

import dns.exception
import dns.name
import dns.ttl

_DELIMITERS = {" ", "\t", "\n", ";", "(", ")", '"'}
_QUOTING_DELIMITERS = {'"'}

EOF = 0
EOL = 1
WHITESPACE = 2
IDENTIFIER = 3
QUOTED_STRING = 4
COMMENT = 5
DELIMITER = 6


class UngetBufferFull(dns.exception.DNSException):
    """An attempt was made to unget a token when the unget buffer was full."""


class Token:
    """A DNS zone file format token.

    ttype: The token type
    value: The token value
    has_escape: Does the token value contain escapes?
    """

    def __init__(
        self,
        ttype: int,
        value: Any = "",
        has_escape: bool = False,
        comment: Optional[str] = None,
    ):
        """Initialize a token instance."""

        self.ttype = ttype
        self.value = value
        self.has_escape = has_escape
        self.comment = comment

    def is_eof(self) -> bool:
        return self.ttype == EOF

    def is_eol(self) -> bool:
        return self.ttype == EOL

    def is_whitespace(self) -> bool:
        return self.ttype == WHITESPACE

    def is_identifier(self) -> bool:
        return self.ttype == IDENTIFIER

    def is_quoted_string(self) -> bool:
        return self.ttype == QUOTED_STRING

    def is_comment(self) -> bool:
        return self.ttype == COMMENT

    def is_delimiter(self) -> bool:  # pragma: no cover (we don't return delimiters yet)
        return self.ttype == DELIMITER

    def is_eol_or_eof(self) -> bool:
        return self.ttype == EOL or self.ttype == EOF

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.ttype == other.ttype and self.value == other.value

    def __ne__(self, other):
        if not isinstance(other, Token):
            return True
        return self.ttype != other.ttype or self.value != other.value

    def __str__(self):
        return '%d "%s"' % (self.ttype, self.value)

    def unescape(self) -> "Token":
        if not self.has_escape:
            return self
        unescaped = ""
        l = len(self.value)
        i = 0
        while i < l:
            c = self.value[i]
            i += 1
            if c == "\\":
                if i >= l:  # pragma: no cover   (can't happen via get())
                    raise dns.exception.UnexpectedEnd
                c = self.value[i]
                i += 1
                if c.isdigit():
                    if i >= l:
                        raise dns.exception.UnexpectedEnd
                    c2 = self.value[i]
                    i += 1
                    if i >= l:
                        raise dns.exception.UnexpectedEnd
                    c3 = self.value[i]
                    i += 1
                    if not (c2.isdigit() and c3.isdigit()):
                        raise dns.exception.SyntaxError
                    codepoint = int(c) * 100 + int(c2) * 10 + int(c3)
                    if codepoint > 255:
                        raise dns.exception.SyntaxError
                    c = chr(codepoint)
            unescaped += c
        return Token(self.ttype, unescaped)

    def unescape_to_bytes(self) -> "Token":
        # We used to use unescape() for TXT-like records, but this
        # caused problems as we'd process DNS escapes into Unicode code
        # points instead of byte values, and then a to_text() of the
        # processed data would not equal the original input.  For
        # example, \226 in the TXT record would have a to_text() of
        # \195\162 because we applied UTF-8 encoding to Unicode code
        # point 226.
        #
        # We now apply escapes while converting directly to bytes,
        # avoiding this double encoding.
        #
        # This code also handles cases where the unicode input has
        # non-ASCII code-points in it by converting it to UTF-8.  TXT
        # records aren't defined for Unicode, but this is the best we
        # can do to preserve meaning.  For example,
        #
        #     foo\u200bbar
        #
        # (where \u200b is Unicode code point 0x200b) will be treated
        # as if the input had been the UTF-8 encoding of that string,
        # namely:
        #
        #     foo\226\128\139bar
        #
        unescaped = b""
        l = len(self.value)
        i = 0
        while i < l:
            c = self.value[i]
            i += 1
            if c == "\\":
                if i >= l:  # pragma: no cover   (can't happen via get())
                    raise dns.exception.UnexpectedEnd
                c = self.value[i]
                i += 1
                if c.isdigit():
                    if i >= l:
                        raise dns.exception.UnexpectedEnd
                    c2 = self.value[i]
                    i += 1
                    if i >= l:
                        raise dns.exception.UnexpectedEnd
                    c3 = self.value[i]
                    i += 1
                    if not (c2.isdigit() and c3.isdigit()):
                        raise dns.exception.SyntaxError
                    codepoint = int(c) * 100 + int(c2) * 10 + int(c3)
                    if codepoint > 255:
                        raise dns.exception.SyntaxError
                    unescaped += b"%c" % (codepoint)
                else:
                    # Note that as mentioned above, if c is a Unicode
                    # code point outside of the ASCII range, then this
                    # += is converting that code point to its UTF-8
                    # encoding and appending multiple bytes to
                    # unescaped.
                    unescaped += c.encode()
            else:
                unescaped += c.encode()
        return Token(self.ttype, bytes(unescaped))


class Tokenizer:
    """A DNS zone file format tokenizer.

    A token object is basically a (type, value) tuple.  The valid
    types are EOF, EOL, WHITESPACE, IDENTIFIER, QUOTED_STRING,
    COMMENT, and DELIMITER.

    file: The file to tokenize

    ungotten_char: The most recently ungotten character, or None.

    ungotten_token: The most recently ungotten token, or None.

    multiline: The current multiline level.  This value is increased
    by one every time a '(' delimiter is read, and decreased by one every time
    a ')' delimiter is read.

    quoting: This variable is true if the tokenizer is currently
    reading a quoted string.

    eof: This variable is true if the tokenizer has encountered EOF.

    delimiters: The current delimiter dictionary.

    line_number: The current line number

    filename: A filename that will be returned by the where() method.

    idna_codec: A dns.name.IDNACodec, specifies the IDNA
    encoder/decoder.  If None, the default IDNA 2003
    encoder/decoder is used.
    """

    def __init__(
        self,
        f: Any = sys.stdin,
        filename: Optional[str] = None,
        idna_codec: Optional[dns.name.IDNACodec] = None,
    ):
        """Initialize a tokenizer instance.

        f: The file to tokenize.  The default is sys.stdin.
        This parameter may also be a string, in which case the tokenizer
        will take its input from the contents of the string.

        filename: the name of the filename that the where() method
        will return.

        idna_codec: A dns.name.IDNACodec, specifies the IDNA
        encoder/decoder.  If None, the default IDNA 2003
        encoder/decoder is used.
        """

        if isinstance(f, str):
            f = io.StringIO(f)
            if filename is None:
                filename = "<string>"
        elif isinstance(f, bytes):
            f = io.StringIO(f.decode())
            if filename is None:
                filename = "<string>"
        else:
            if filename is None:
                if f is sys.stdin:
                    filename = "<stdin>"
                else:
                    filename = "<file>"
        self.file = f
        self.ungotten_char: Optional[str] = None
        self.ungotten_token: Optional[Token] = None
        self.multiline = 0
        self.quoting = False
        self.eof = False
        self.delimiters = _DELIMITERS
        self.line_number = 1
        assert filename is not None
        self.filename = filename
        if idna_codec is None:
            self.idna_codec: dns.name.IDNACodec = dns.name.IDNA_2003
        else:
            self.idna_codec = idna_codec

    def _get_char(self) -> str:
        """Read a character from input."""

        if self.ungotten_char is None:
            if self.eof:
                c = ""
            else:
                c = self.file.read(1)
                if c == "":
                    self.eof = True
                elif c == "\n":
                    self.line_number += 1
        else:
            c = self.ungotten_char
            self.ungotten_char = None
        return c

    def where(self) -> Tuple[str, int]:
        """Return the current location in the input.

        Returns a (string, int) tuple.  The first item is the filename of
        the input, the second is the current line number.
        """

        return (self.filename, self.line_number)

    def _unget_char(self, c: str) -> None:
        """Unget a character.

        The unget buffer for characters is only one character large; it is
        an error to try to unget a character when the unget buffer is not
        empty.

        c: the character to unget
        raises UngetBufferFull: there is already an ungotten char
        """

        if self.ungotten_char is not None:
            # this should never happen!
            raise UngetBufferFull  # pragma: no cover
        self.ungotten_char = c

    def skip_whitespace(self) -> int:
        """Consume input until a non-whitespace character is encountered.

        The non-whitespace character is then ungotten, and the number of
        whitespace characters consumed is returned.

        If the tokenizer is in multiline mode, then newlines are whitespace.

        Returns the number of characters skipped.
        """

        skipped = 0
        while True:
            c = self._get_char()
            if c != " " and c != "\t":
                if (c != "\n") or not self.multiline:
                    self._unget_char(c)
                    return skipped
            skipped += 1

    def get(self, want_leading: bool = False, want_comment: bool = False) -> Token:
        """Get the next token.

        want_leading: If True, return a WHITESPACE token if the
        first character read is whitespace.  The default is False.

        want_comment: If True, return a COMMENT token if the
        first token read is a comment.  The default is False.

        Raises dns.exception.UnexpectedEnd: input ended prematurely

        Raises dns.exception.SyntaxError: input was badly formed

        Returns a Token.
        """

        if self.ungotten_token is not None:
            utoken = self.ungotten_token
            self.ungotten_token = None
            if utoken.is_whitespace():
                if want_leading:
                    return utoken
            elif utoken.is_comment():
                if want_comment:
                    return utoken
            else:
                return utoken
        skipped = self.skip_whitespace()
        if want_leading and skipped > 0:
            return Token(WHITESPACE, " ")
        token = ""
        ttype = IDENTIFIER
        has_escape = False
        while True:
            c = self._get_char()
            if c == "" or c in self.delimiters:
                if c == "" and self.quoting:
                    raise dns.exception.UnexpectedEnd
                if token == "" and ttype != QUOTED_STRING:
                    if c == "(":
                        self.multiline += 1
                        self.skip_whitespace()
                        continue
                    elif c == ")":
                        if self.multiline <= 0:
                            raise dns.exception.SyntaxError
                        self.multiline -= 1
                        self.skip_whitespace()
                        continue
                    elif c == '"':
                        if not self.quoting:
                            self.quoting = True
                            self.delimiters = _QUOTING_DELIMITERS
                            ttype = QUOTED_STRING
                            continue
                        else:
                            self.quoting = False
                            self.delimiters = _DELIMITERS
                            self.skip_whitespace()
                            continue
                    elif c == "\n":
                        return Token(EOL, "\n")
                    elif c == ";":
                        while 1:
                            c = self._get_char()
                            if c == "\n" or c == "":
                                break
                            token += c
                        if want_comment:
                            self._unget_char(c)
                            return Token(COMMENT, token)
                        elif c == "":
                            if self.multiline:
                                raise dns.exception.SyntaxError(
                                    "unbalanced parentheses"
                                )
                            return Token(EOF, comment=token)
                        elif self.multiline:
                            self.skip_whitespace()
                            token = ""
                            continue
                        else:
                            return Token(EOL, "\n", comment=token)
                    else:
                        # This code exists in case we ever want a
                        # delimiter to be returned.  It never produces
                        # a token currently.
                        token = c
                        ttype = DELIMITER
                else:
                    self._unget_char(c)
                break
            elif self.quoting and c == "\n":
                raise dns.exception.SyntaxError("newline in quoted string")
            elif c == "\\":
                #
                # It's an escape.  Put it and the next character into
                # the token; it will be checked later for goodness.
                #
                token += c
                has_escape = True
                c = self._get_char()
                if c == "" or (c == "\n" and not self.quoting):
                    raise dns.exception.UnexpectedEnd
            token += c
        if token == "" and ttype != QUOTED_STRING:
            if self.multiline:
                raise dns.exception.SyntaxError("unbalanced parentheses")
            ttype = EOF
        return Token(ttype, token, has_escape)

    def unget(self, token: Token) -> None:
        """Unget a token.

        The unget buffer for tokens is only one token large; it is
        an error to try to unget a token when the unget buffer is not
        empty.

        token: the token to unget

        Raises UngetBufferFull: there is already an ungotten token
        """

        if self.ungotten_token is not None:
            raise UngetBufferFull
        self.ungotten_token = token

    def next(self):
        """Return the next item in an iteration.

        Returns a Token.
        """

        token = self.get()
        if token.is_eof():
            raise StopIteration
        return token

    __next__ = next

    def __iter__(self):
        return self

    # Helpers

    def get_int(self, base: int = 10) -> int:
        """Read the next token and interpret it as an unsigned integer.

        Raises dns.exception.SyntaxError if not an unsigned integer.

        Returns an int.
        """

        token = self.get().unescape()
        if not token.is_identifier():
            raise dns.exception.SyntaxError("expecting an identifier")
        if not token.value.isdigit():
            raise dns.exception.SyntaxError("expecting an integer")
        return int(token.value, base)

    def get_uint8(self) -> int:
        """Read the next token and interpret it as an 8-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not an 8-bit unsigned integer.

        Returns an int.
        """

        value = self.get_int()
        if value < 0 or value > 255:
            raise dns.exception.SyntaxError(
                "%d is not an unsigned 8-bit integer" % value
            )
        return value

    def get_uint16(self, base: int = 10) -> int:
        """Read the next token and interpret it as a 16-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not a 16-bit unsigned integer.

        Returns an int.
        """

        value = self.get_int(base=base)
        if value < 0 or value > 65535:
            if base == 8:
                raise dns.exception.SyntaxError(
                    "%o is not an octal unsigned 16-bit integer" % value
                )
            else:
                raise dns.exception.SyntaxError(
                    "%d is not an unsigned 16-bit integer" % value
                )
        return value

    def get_uint32(self, base: int = 10) -> int:
        """Read the next token and interpret it as a 32-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not a 32-bit unsigned integer.

        Returns an int.
        """

        value = self.get_int(base=base)
        if value < 0 or value > 4294967295:
            raise dns.exception.SyntaxError(
                "%d is not an unsigned 32-bit integer" % value
            )
        return value

    def get_uint48(self, base: int = 10) -> int:
        """Read the next token and interpret it as a 48-bit unsigned
        integer.

        Raises dns.exception.SyntaxError if not a 48-bit unsigned integer.

        Returns an int.
        """

        value = self.get_int(base=base)
        if value < 0 or value > 281474976710655:
            raise dns.exception.SyntaxError(
                "%d is not an unsigned 48-bit integer" % value
            )
        return value

    def get_string(self, max_length: Optional[int] = None) -> str:
        """Read the next token and interpret it as a string.

        Raises dns.exception.SyntaxError if not a string.
        Raises dns.exception.SyntaxError if token value length
        exceeds max_length (if specified).

        Returns a string.
        """

        token = self.get().unescape()
        if not (token.is_identifier() or token.is_quoted_string()):
            raise dns.exception.SyntaxError("expecting a string")
        if max_length and len(token.value) > max_length:
            raise dns.exception.SyntaxError("string too long")
        return token.value

    def get_identifier(self) -> str:
        """Read the next token, which should be an identifier.

        Raises dns.exception.SyntaxError if not an identifier.

        Returns a string.
        """

        token = self.get().unescape()
        if not token.is_identifier():
            raise dns.exception.SyntaxError("expecting an identifier")
        return token.value

    def get_remaining(self, max_tokens: Optional[int] = None) -> List[Token]:
        """Return the remaining tokens on the line, until an EOL or EOF is seen.

        max_tokens: If not None, stop after this number of tokens.

        Returns a list of tokens.
        """

        tokens = []
        while True:
            token = self.get()
            if token.is_eol_or_eof():
                self.unget(token)
                break
            tokens.append(token)
            if len(tokens) == max_tokens:
                break
        return tokens

    def concatenate_remaining_identifiers(self, allow_empty: bool = False) -> str:
        """Read the remaining tokens on the line, which should be identifiers.

        Raises dns.exception.SyntaxError if there are no remaining tokens,
        unless `allow_empty=True` is given.

        Raises dns.exception.SyntaxError if a token is seen that is not an
        identifier.

        Returns a string containing a concatenation of the remaining
        identifiers.
        """
        s = ""
        while True:
            token = self.get().unescape()
            if token.is_eol_or_eof():
                self.unget(token)
                break
            if not token.is_identifier():
                raise dns.exception.SyntaxError
            s += token.value
        if not (allow_empty or s):
            raise dns.exception.SyntaxError("expecting another identifier")
        return s

    def as_name(
        self,
        token: Token,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = False,
        relativize_to: Optional[dns.name.Name] = None,
    ) -> dns.name.Name:
        """Try to interpret the token as a DNS name.

        Raises dns.exception.SyntaxError if not a name.

        Returns a dns.name.Name.
        """
        if not token.is_identifier():
            raise dns.exception.SyntaxError("expecting an identifier")
        name = dns.name.from_text(token.value, origin, self.idna_codec)
        return name.choose_relativity(relativize_to or origin, relativize)

    def get_name(
        self,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = False,
        relativize_to: Optional[dns.name.Name] = None,
    ) -> dns.name.Name:
        """Read the next token and interpret it as a DNS name.

        Raises dns.exception.SyntaxError if not a name.

        Returns a dns.name.Name.
        """

        token = self.get()
        return self.as_name(token, origin, relativize, relativize_to)

    def get_eol_as_token(self) -> Token:
        """Read the next token and raise an exception if it isn't EOL or
        EOF.

        Returns a string.
        """

        token = self.get()
        if not token.is_eol_or_eof():
            raise dns.exception.SyntaxError(
                'expected EOL or EOF, got %d "%s"' % (token.ttype, token.value)
            )
        return token

    def get_eol(self) -> str:
        return self.get_eol_as_token().value

    def get_ttl(self) -> int:
        """Read the next token and interpret it as a DNS TTL.

        Raises dns.exception.SyntaxError or dns.ttl.BadTTL if not an
        identifier or badly formed.

        Returns an int.
        """

        token = self.get().unescape()
        if not token.is_identifier():
            raise dns.exception.SyntaxError("expecting an identifier")
        return dns.ttl.from_text(token.value)
