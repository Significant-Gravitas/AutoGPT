# Copyright (c) 2000 David Abrahams. Permission to copy, use, modify, sell
# and distribute this software is granted provided this copyright
# notice appears in all copies. This software is provided "as is" without
# express or implied warranty, and with no claim as to its suitability for
# any purpose.
"""Provides a class Stdin which can be used to emulate the regular old
sys.stdin for the PythonWin interactive window. Right now it just pops
up a raw_input() dialog. With luck, someone will integrate it into the
actual PythonWin interactive window someday.

WARNING: Importing this file automatically replaces sys.stdin with an
instance of Stdin (below). This is useful because you can just open
Stdin.py in PythonWin and hit the import button to get it set up right
if you don't feel like changing PythonWin's source. To put things back
the way they were, simply use this magic incantation:
    import sys
    sys.stdin = sys.stdin.real_file
"""
import sys

try:
    get_input_line = raw_input  # py2x
except NameError:
    get_input_line = input  # py3k


class Stdin:
    def __init__(self):
        self.real_file = sys.stdin  # NOTE: Likely to be None in py3k
        self.buffer = ""
        self.closed = False

    def __getattr__(self, name):
        """Forward most functions to the real sys.stdin for absolute realism."""
        if self.real_file is None:
            raise AttributeError(name)
        return getattr(self.real_file, name)

    def isatty(self):
        """Return 1 if the file is connected to a tty(-like) device, else 0."""
        return 1

    def read(self, size=-1):
        """Read at most size bytes from the file (less if the read
        hits EOF or no more data is immediately available on a pipe,
        tty or similar device). If the size argument is negative or
        omitted, read all data until EOF is reached. The bytes are
        returned as a string object. An empty string is returned when
        EOF is encountered immediately. (For certain files, like ttys,
        it makes sense to continue reading after an EOF is hit.)"""
        result_size = self.__get_lines(size)
        return self.__extract_from_buffer(result_size)

    def readline(self, size=-1):
        """Read one entire line from the file. A trailing newline
        character is kept in the string2.6 (but may be absent when a file ends
        with an incomplete line). If the size argument is present and
        non-negative, it is a maximum byte count (including the trailing
        newline) and an incomplete line may be returned. An empty string is
        returned when EOF is hit immediately. Note: unlike stdio's fgets(),
        the returned string contains null characters ('\0') if they occurred
        in the input.
        """
        maximum_result_size = self.__get_lines(size, lambda buffer: "\n" in buffer)

        if "\n" in self.buffer[:maximum_result_size]:
            result_size = self.buffer.find("\n", 0, maximum_result_size) + 1
            assert result_size > 0
        else:
            result_size = maximum_result_size

        return self.__extract_from_buffer(result_size)

    def __extract_from_buffer(self, character_count):
        """Remove the first character_count characters from the internal buffer and
        return them.
        """
        result = self.buffer[:character_count]
        self.buffer = self.buffer[character_count:]
        return result

    def __get_lines(self, desired_size, done_reading=lambda buffer: False):
        """Keep adding lines to our internal buffer until done_reading(self.buffer)
        is true or EOF has been reached or we have desired_size bytes in the buffer.
        If desired_size < 0, we are never satisfied until we reach EOF. If done_reading
        is not supplied, it is not consulted.

        If desired_size < 0, returns the length of the internal buffer. Otherwise,
        returns desired_size.
        """
        while not done_reading(self.buffer) and (
            desired_size < 0 or len(self.buffer) < desired_size
        ):
            try:
                self.__get_line()
            except (
                EOFError,
                KeyboardInterrupt,
            ):  # deal with cancellation of get_input_line dialog
                desired_size = len(self.buffer)  # Be satisfied!

        if desired_size < 0:
            return len(self.buffer)
        else:
            return desired_size

    def __get_line(self):
        """Grab one line from get_input_line() and append it to the buffer."""
        line = get_input_line()
        print(">>>", line)  # echo input to console
        self.buffer = self.buffer + line + "\n"

    def readlines(self, *sizehint):
        """Read until EOF using readline() and return a list containing the lines
        thus read. If the optional sizehint argument is present, instead of
        reading up to EOF, whole lines totalling approximately sizehint bytes
        (possibly after rounding up to an internal buffer size) are read.
        """
        result = []
        total_read = 0
        while sizehint == () or total_read < sizehint[0]:
            line = self.readline()
            if line == "":
                break
            total_read = total_read + len(line)
            result.append(line)
        return result


if __name__ == "__main__":
    test_input = r"""this is some test
input that I am hoping
~
will be very instructive
and when I am done
I will have tested everything.
Twelve and twenty blackbirds
baked in a pie. Patty cake
patty cake so am I.
~
Thirty-five niggling idiots!
Sell you soul to the devil, baby
"""

    def fake_raw_input(prompt=None):
        """Replacement for raw_input() which pulls lines out of global test_input.
        For testing only!
        """
        global test_input
        if "\n" not in test_input:
            end_of_line_pos = len(test_input)
        else:
            end_of_line_pos = test_input.find("\n")
        result = test_input[:end_of_line_pos]
        test_input = test_input[end_of_line_pos + 1 :]
        if len(result) == 0 or result[0] == "~":
            raise EOFError()
        return result

    get_input_line = fake_raw_input

    # Some completely inadequate tests, just to make sure the code's not totally broken
    try:
        x = Stdin()
        print(x.read())
        print(x.readline())
        print(x.read(12))
        print(x.readline(47))
        print(x.readline(3))
        print(x.readlines())
    finally:
        get_input_line = raw_input
else:
    import sys

    sys.stdin = Stdin()
