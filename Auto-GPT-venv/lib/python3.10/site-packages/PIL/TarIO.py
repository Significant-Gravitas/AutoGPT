#
# The Python Imaging Library.
# $Id$
#
# read files from within a tar file
#
# History:
# 95-06-18 fl   Created
# 96-05-28 fl   Open files in binary mode
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1995-96.
#
# See the README file for information on usage and redistribution.
#

import io

from . import ContainerIO


class TarIO(ContainerIO.ContainerIO):
    """A file object that provides read access to a given member of a TAR file."""

    def __init__(self, tarfile, file):
        """
        Create file object.

        :param tarfile: Name of TAR file.
        :param file: Name of member file.
        """
        self.fh = open(tarfile, "rb")

        while True:
            s = self.fh.read(512)
            if len(s) != 512:
                msg = "unexpected end of tar file"
                raise OSError(msg)

            name = s[:100].decode("utf-8")
            i = name.find("\0")
            if i == 0:
                msg = "cannot find subfile"
                raise OSError(msg)
            if i > 0:
                name = name[:i]

            size = int(s[124:135], 8)

            if file == name:
                break

            self.fh.seek((size + 511) & (~511), io.SEEK_CUR)

        # Open region
        super().__init__(self.fh, self.fh.tell(), size)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.fh.close()
