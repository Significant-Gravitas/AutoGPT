# cmdline - command line utilities.
import string
import sys

import win32ui


def ParseArgs(str):
    import string

    ret = []
    pos = 0
    length = len(str)
    while pos < length:
        try:
            while str[pos] in string.whitespace:
                pos = pos + 1
        except IndexError:
            break
        if pos >= length:
            break
        if str[pos] == '"':
            pos = pos + 1
            try:
                endPos = str.index('"', pos) - 1
                nextPos = endPos + 2
            except ValueError:
                endPos = length
                nextPos = endPos + 1
        else:
            endPos = pos
            while endPos < length and not str[endPos] in string.whitespace:
                endPos = endPos + 1
            nextPos = endPos + 1
        ret.append(str[pos : endPos + 1].strip())
        pos = nextPos
    return ret


def FixArgFileName(fileName):
    """Convert a filename on the commandline to something useful.
    Given an automatic filename on the commandline, turn it a python module name,
    with the path added to sys.path."""
    import os

    path, fname = os.path.split(fileName)
    if len(path) == 0:
        path = os.curdir
    path = os.path.abspath(path)
    # must check that the command line arg's path is in sys.path
    for syspath in sys.path:
        if os.path.abspath(syspath) == path:
            break
    else:
        sys.path.append(path)
    return os.path.splitext(fname)[0]
