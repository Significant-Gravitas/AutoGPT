"""
Script which takes one or more file paths and reports on their detected
encodings

Example::

    % chardetect somefile someotherfile
    somefile: windows-1252 with confidence 0.5
    someotherfile: ascii with confidence 1.0

If no paths are provided, it takes its input from stdin.

"""


import argparse
import sys

from .. import __version__
from ..universaldetector import UniversalDetector


def description_of(lines, name="stdin"):
    """
    Return a string describing the probable encoding of a file or
    list of strings.

    :param lines: The lines to get the encoding of.
    :type lines: Iterable of bytes
    :param name: Name of file or collection of lines
    :type name: str
    """
    u = UniversalDetector()
    for line in lines:
        line = bytearray(line)
        u.feed(line)
        # shortcut out of the loop to save reading further - particularly useful if we read a BOM.
        if u.done:
            break
    u.close()
    result = u.result
    if result["encoding"]:
        return f'{name}: {result["encoding"]} with confidence {result["confidence"]}'
    return f"{name}: no result"


def main(argv=None):
    """
    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    """
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Takes one or more file paths and reports their detected \
                     encodings"
    )
    parser.add_argument(
        "input",
        help="File whose encoding we would like to determine. \
                              (default: stdin)",
        type=argparse.FileType("rb"),
        nargs="*",
        default=[sys.stdin.buffer],
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args(argv)

    for f in args.input:
        if f.isatty():
            print(
                "You are running chardetect interactively. Press "
                "CTRL-D twice at the start of a blank line to signal the "
                "end of your input. If you want help, run chardetect "
                "--help\n",
                file=sys.stderr,
            )
        print(description_of(f, f.name))


if __name__ == "__main__":
    main()
