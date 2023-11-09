"""
_url.py
websocket - WebSocket client library for Python

Copyright 2022 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
__all__ = ["NoLock", "validate_utf8", "extract_err_message", "extract_error_code"]


class NoLock:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


try:
    # If wsaccel is available we use compiled routines to validate UTF-8
    # strings.
    from wsaccel.utf8validator import Utf8Validator

    def _validate_utf8(utfbytes):
        return Utf8Validator().validate(utfbytes)[0]

except ImportError:
    # UTF-8 validator
    # python implementation of http://bjoern.hoehrmann.de/utf-8/decoder/dfa/

    _UTF8_ACCEPT = 0
    _UTF8_REJECT = 12

    _UTF8D = [
        # The first part of the table maps bytes to character classes that
        # to reduce the size of the transition table and create bitmasks.
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
        7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
        8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        10,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3, 11,6,6,6,5,8,8,8,8,8,8,8,8,8,8,8,

        # The second part is a transition table that maps a combination
        # of a state of the automaton and a character class to a state.
        0,12,24,36,60,96,84,12,12,12,48,72, 12,12,12,12,12,12,12,12,12,12,12,12,
        12, 0,12,12,12,12,12, 0,12, 0,12,12, 12,24,12,12,12,12,12,24,12,24,12,12,
        12,12,12,12,12,12,12,24,12,12,12,12, 12,24,12,12,12,12,12,12,12,24,12,12,
        12,12,12,12,12,12,12,36,12,36,12,12, 12,36,12,12,12,12,12,36,12,36,12,12,
        12,36,12,12,12,12,12,12,12,12,12,12, ]

    def _decode(state, codep, ch):
        tp = _UTF8D[ch]

        codep = (ch & 0x3f) | (codep << 6) if (
            state != _UTF8_ACCEPT) else (0xff >> tp) & ch
        state = _UTF8D[256 + state + tp]

        return state, codep

    def _validate_utf8(utfbytes):
        state = _UTF8_ACCEPT
        codep = 0
        for i in utfbytes:
            state, codep = _decode(state, codep, i)
            if state == _UTF8_REJECT:
                return False

        return True


def validate_utf8(utfbytes):
    """
    validate utf8 byte string.
    utfbytes: utf byte string to check.
    return value: if valid utf8 string, return true. Otherwise, return false.
    """
    return _validate_utf8(utfbytes)


def extract_err_message(exception):
    if exception.args:
        return exception.args[0]
    else:
        return None


def extract_error_code(exception):
    if exception.args and len(exception.args) > 1:
        return exception.args[0] if isinstance(exception.args[0], int) else None
