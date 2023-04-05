# -*- coding: utf-8 -*-
from gtts.tokenizer.symbols import ALL_PUNC as punc
from string import whitespace as ws
import re

_ALL_PUNC_OR_SPACE = re.compile(u"^[{}]*$".format(re.escape(punc + ws)))
"""Regex that matches if an entire line is only comprised
of whitespace and punctuation

"""


def _minimize(the_string, delim, max_size):
    """Recursively split a string in the largest chunks
    possible from the highest position of a delimiter all the way
    to a maximum size

    Args:
        the_string (string): The string to split.
        delim (string): The delimiter to split on.
        max_size (int): The maximum size of a chunk.

    Returns:
        list: the minimized string in tokens

    Every chunk size will be at minimum ``the_string[0:idx]`` where ``idx``
    is the highest index of ``delim`` found in ``the_string``; and at maximum
    ``the_string[0:max_size]`` if no ``delim`` was found in ``the_string``.
    In the latter case, the split will occur at ``the_string[max_size]``
    which can be any character. The function runs itself again on the rest of
    ``the_string`` (``the_string[idx:]``) until no chunk is larger than
    ``max_size``.

    """
    # Remove `delim` from start of `the_string`
    # i.e. prevent a recursive infinite loop on `the_string[0:0]`
    # if `the_string` starts with `delim` and is larger than `max_size`
    if the_string.startswith(delim):
        the_string = the_string[_len(delim) :]

    if _len(the_string) > max_size:
        try:
            # Find the highest index of `delim` in `the_string[0:max_size]`
            # i.e. `the_string` will be cut in half on `delim` index
            idx = the_string.rindex(delim, 0, max_size)
        except ValueError:
            # `delim` not found in `the_string`, index becomes `max_size`
            # i.e. `the_string` will be cut in half arbitrarily on `max_size`
            idx = max_size
        # Call itself again for `the_string[idx:]`
        return [the_string[:idx]] + _minimize(the_string[idx:], delim, max_size)
    else:
        return [the_string]


def _len(text):
    """Same as ``len(text)`` for a string but that decodes
    ``text`` first in Python 2.x

    Args:
        text (string): String to get the size of.

    Returns:
        int: The size of the string.
    """
    try:
        # Python 2
        return len(unicode(text))
    except NameError:  # pragma: no cover
        # Python 3
        return len(text)


def _clean_tokens(tokens):
    """Clean a list of strings

    Args:
        tokens (list): A list of strings (tokens) to clean.

    Returns:
        list: Stripped strings ``tokens`` without the original elements
            that only consisted of whitespace and/or punctuation characters.

    """
    return [t.strip() for t in tokens if not _ALL_PUNC_OR_SPACE.match(t)]


def _translate_url(tld="com", path=""):
    """Generates a Google Translate URL

    Args:
        tld (string): Top-level domain for the Google Translate host,
            i.e ``https://translate.google.<tld>``. Default is ``com``.
        path: (string): A path to append to the Google Translate host,
            i.e ``https://translate.google.com/<path>``. Default is ``""``.

    Returns:
        string: A Google Translate URL `https://translate.google.<tld>/path`
    """
    _GOOGLE_TTS_URL = "https://translate.google.{}/{}"
    return _GOOGLE_TTS_URL.format(tld, path)
