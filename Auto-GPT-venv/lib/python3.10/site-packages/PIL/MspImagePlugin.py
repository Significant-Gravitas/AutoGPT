#
# The Python Imaging Library.
#
# MSP file handling
#
# This is the format used by the Paint program in Windows 1 and 2.
#
# History:
#       95-09-05 fl     Created
#       97-01-03 fl     Read/write MSP images
#       17-02-21 es     Fixed RLE interpretation
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1995-97.
# Copyright (c) Eric Soroos 2017.
#
# See the README file for information on usage and redistribution.
#
# More info on this format: https://archive.org/details/gg243631
# Page 313:
# Figure 205. Windows Paint Version 1: "DanM" Format
# Figure 206. Windows Paint Version 2: "LinS" Format. Used in Windows V2.03
#
# See also: https://www.fileformat.info/format/mspaint/egff.htm

import io
import struct

from . import Image, ImageFile
from ._binary import i16le as i16
from ._binary import o16le as o16

#
# read MSP files


def _accept(prefix):
    return prefix[:4] in [b"DanM", b"LinS"]


##
# Image plugin for Windows MSP images.  This plugin supports both
# uncompressed (Windows 1.0).


class MspImageFile(ImageFile.ImageFile):
    format = "MSP"
    format_description = "Windows Paint"

    def _open(self):
        # Header
        s = self.fp.read(32)
        if not _accept(s):
            msg = "not an MSP file"
            raise SyntaxError(msg)

        # Header checksum
        checksum = 0
        for i in range(0, 32, 2):
            checksum = checksum ^ i16(s, i)
        if checksum != 0:
            msg = "bad MSP checksum"
            raise SyntaxError(msg)

        self.mode = "1"
        self._size = i16(s, 4), i16(s, 6)

        if s[:4] == b"DanM":
            self.tile = [("raw", (0, 0) + self.size, 32, ("1", 0, 1))]
        else:
            self.tile = [("MSP", (0, 0) + self.size, 32, None)]


class MspDecoder(ImageFile.PyDecoder):
    # The algo for the MSP decoder is from
    # https://www.fileformat.info/format/mspaint/egff.htm
    # cc-by-attribution -- That page references is taken from the
    # Encyclopedia of Graphics File Formats and is licensed by
    # O'Reilly under the Creative Common/Attribution license
    #
    # For RLE encoded files, the 32byte header is followed by a scan
    # line map, encoded as one 16bit word of encoded byte length per
    # line.
    #
    # NOTE: the encoded length of the line can be 0. This was not
    # handled in the previous version of this encoder, and there's no
    # mention of how to handle it in the documentation. From the few
    # examples I've seen, I've assumed that it is a fill of the
    # background color, in this case, white.
    #
    #
    # Pseudocode of the decoder:
    # Read a BYTE value as the RunType
    #  If the RunType value is zero
    #   Read next byte as the RunCount
    #   Read the next byte as the RunValue
    #   Write the RunValue byte RunCount times
    #  If the RunType value is non-zero
    #   Use this value as the RunCount
    #   Read and write the next RunCount bytes literally
    #
    #  e.g.:
    #  0x00 03 ff 05 00 01 02 03 04
    #  would yield the bytes:
    #  0xff ff ff 00 01 02 03 04
    #
    # which are then interpreted as a bit packed mode '1' image

    _pulls_fd = True

    def decode(self, buffer):
        img = io.BytesIO()
        blank_line = bytearray((0xFF,) * ((self.state.xsize + 7) // 8))
        try:
            self.fd.seek(32)
            rowmap = struct.unpack_from(
                f"<{self.state.ysize}H", self.fd.read(self.state.ysize * 2)
            )
        except struct.error as e:
            msg = "Truncated MSP file in row map"
            raise OSError(msg) from e

        for x, rowlen in enumerate(rowmap):
            try:
                if rowlen == 0:
                    img.write(blank_line)
                    continue
                row = self.fd.read(rowlen)
                if len(row) != rowlen:
                    msg = f"Truncated MSP file, expected {rowlen} bytes on row {x}"
                    raise OSError(msg)
                idx = 0
                while idx < rowlen:
                    runtype = row[idx]
                    idx += 1
                    if runtype == 0:
                        (runcount, runval) = struct.unpack_from("Bc", row, idx)
                        img.write(runval * runcount)
                        idx += 2
                    else:
                        runcount = runtype
                        img.write(row[idx : idx + runcount])
                        idx += runcount

            except struct.error as e:
                msg = f"Corrupted MSP file in row {x}"
                raise OSError(msg) from e

        self.set_as_raw(img.getvalue(), ("1", 0, 1))

        return -1, 0


Image.register_decoder("MSP", MspDecoder)


#
# write MSP files (uncompressed only)


def _save(im, fp, filename):
    if im.mode != "1":
        msg = f"cannot write mode {im.mode} as MSP"
        raise OSError(msg)

    # create MSP header
    header = [0] * 16

    header[0], header[1] = i16(b"Da"), i16(b"nM")  # version 1
    header[2], header[3] = im.size
    header[4], header[5] = 1, 1
    header[6], header[7] = 1, 1
    header[8], header[9] = im.size

    checksum = 0
    for h in header:
        checksum = checksum ^ h
    header[12] = checksum  # FIXME: is this the right field?

    # header
    for h in header:
        fp.write(o16(h))

    # image body
    ImageFile._save(im, fp, [("raw", (0, 0) + im.size, 32, ("1", 0, 1))])


#
# registry

Image.register_open(MspImageFile.format, MspImageFile, _accept)
Image.register_save(MspImageFile.format, _save)

Image.register_extension(MspImageFile.format, ".msp")
