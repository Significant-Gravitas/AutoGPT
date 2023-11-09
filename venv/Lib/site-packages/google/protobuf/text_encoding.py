# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Encoding related utilities."""
import re

_cescape_chr_to_symbol_map = {}
_cescape_chr_to_symbol_map[9] = r'\t'  # optional escape
_cescape_chr_to_symbol_map[10] = r'\n'  # optional escape
_cescape_chr_to_symbol_map[13] = r'\r'  # optional escape
_cescape_chr_to_symbol_map[34] = r'\"'  # necessary escape
_cescape_chr_to_symbol_map[39] = r"\'"  # optional escape
_cescape_chr_to_symbol_map[92] = r'\\'  # necessary escape

# Lookup table for unicode
_cescape_unicode_to_str = [chr(i) for i in range(0, 256)]
for byte, string in _cescape_chr_to_symbol_map.items():
  _cescape_unicode_to_str[byte] = string

# Lookup table for non-utf8, with necessary escapes at (o >= 127 or o < 32)
_cescape_byte_to_str = ([r'\%03o' % i for i in range(0, 32)] +
                        [chr(i) for i in range(32, 127)] +
                        [r'\%03o' % i for i in range(127, 256)])
for byte, string in _cescape_chr_to_symbol_map.items():
  _cescape_byte_to_str[byte] = string
del byte, string


def CEscape(text, as_utf8) -> str:
  """Escape a bytes string for use in an text protocol buffer.

  Args:
    text: A byte string to be escaped.
    as_utf8: Specifies if result may contain non-ASCII characters.
        In Python 3 this allows unescaped non-ASCII Unicode characters.
        In Python 2 the return value will be valid UTF-8 rather than only ASCII.
  Returns:
    Escaped string (str).
  """
  # Python's text.encode() 'string_escape' or 'unicode_escape' codecs do not
  # satisfy our needs; they encodes unprintable characters using two-digit hex
  # escapes whereas our C++ unescaping function allows hex escapes to be any
  # length.  So, "\0011".encode('string_escape') ends up being "\\x011", which
  # will be decoded in C++ as a single-character string with char code 0x11.
  text_is_unicode = isinstance(text, str)
  if as_utf8 and text_is_unicode:
    # We're already unicode, no processing beyond control char escapes.
    return text.translate(_cescape_chr_to_symbol_map)
  ord_ = ord if text_is_unicode else lambda x: x  # bytes iterate as ints.
  if as_utf8:
    return ''.join(_cescape_unicode_to_str[ord_(c)] for c in text)
  return ''.join(_cescape_byte_to_str[ord_(c)] for c in text)


_CUNESCAPE_HEX = re.compile(r'(\\+)x([0-9a-fA-F])(?![0-9a-fA-F])')


def CUnescape(text: str) -> bytes:
  """Unescape a text string with C-style escape sequences to UTF-8 bytes.

  Args:
    text: The data to parse in a str.
  Returns:
    A byte string.
  """

  def ReplaceHex(m):
    # Only replace the match if the number of leading back slashes is odd. i.e.
    # the slash itself is not escaped.
    if len(m.group(1)) & 1:
      return m.group(1) + 'x0' + m.group(2)
    return m.group(0)

  # This is required because the 'string_escape' encoding doesn't
  # allow single-digit hex escapes (like '\xf').
  result = _CUNESCAPE_HEX.sub(ReplaceHex, text)

  return (result.encode('utf-8')  # Make it bytes to allow decode.
          .decode('unicode_escape')
          # Make it bytes again to return the proper type.
          .encode('raw_unicode_escape'))
