#
# The Python Imaging Library.
# $Id$
#
# Sun image file handling
#
# History:
# 1995-09-10 fl   Created
# 1996-05-28 fl   Fixed 32-bit alignment
# 1998-12-29 fl   Import ImagePalette module
# 2001-12-18 fl   Fixed palette loading (from Jean-Claude Rimbault)
#
# Copyright (c) 1997-2001 by Secret Labs AB
# Copyright (c) 1995-1996 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#


from . import Image, ImageFile, ImagePalette
from ._binary import i32be as i32


def _accept(prefix):
    return len(prefix) >= 4 and i32(prefix) == 0x59A66A95


##
# Image plugin for Sun raster files.


class SunImageFile(ImageFile.ImageFile):
    format = "SUN"
    format_description = "Sun Raster File"

    def _open(self):
        # The Sun Raster file header is 32 bytes in length
        # and has the following format:

        #     typedef struct _SunRaster
        #     {
        #         DWORD MagicNumber;      /* Magic (identification) number */
        #         DWORD Width;            /* Width of image in pixels */
        #         DWORD Height;           /* Height of image in pixels */
        #         DWORD Depth;            /* Number of bits per pixel */
        #         DWORD Length;           /* Size of image data in bytes */
        #         DWORD Type;             /* Type of raster file */
        #         DWORD ColorMapType;     /* Type of color map */
        #         DWORD ColorMapLength;   /* Size of the color map in bytes */
        #     } SUNRASTER;

        # HEAD
        s = self.fp.read(32)
        if not _accept(s):
            msg = "not an SUN raster file"
            raise SyntaxError(msg)

        offset = 32

        self._size = i32(s, 4), i32(s, 8)

        depth = i32(s, 12)
        # data_length = i32(s, 16)   # unreliable, ignore.
        file_type = i32(s, 20)
        palette_type = i32(s, 24)  # 0: None, 1: RGB, 2: Raw/arbitrary
        palette_length = i32(s, 28)

        if depth == 1:
            self.mode, rawmode = "1", "1;I"
        elif depth == 4:
            self.mode, rawmode = "L", "L;4"
        elif depth == 8:
            self.mode = rawmode = "L"
        elif depth == 24:
            if file_type == 3:
                self.mode, rawmode = "RGB", "RGB"
            else:
                self.mode, rawmode = "RGB", "BGR"
        elif depth == 32:
            if file_type == 3:
                self.mode, rawmode = "RGB", "RGBX"
            else:
                self.mode, rawmode = "RGB", "BGRX"
        else:
            msg = "Unsupported Mode/Bit Depth"
            raise SyntaxError(msg)

        if palette_length:
            if palette_length > 1024:
                msg = "Unsupported Color Palette Length"
                raise SyntaxError(msg)

            if palette_type != 1:
                msg = "Unsupported Palette Type"
                raise SyntaxError(msg)

            offset = offset + palette_length
            self.palette = ImagePalette.raw("RGB;L", self.fp.read(palette_length))
            if self.mode == "L":
                self.mode = "P"
                rawmode = rawmode.replace("L", "P")

        # 16 bit boundaries on stride
        stride = ((self.size[0] * depth + 15) // 16) * 2

        # file type: Type is the version (or flavor) of the bitmap
        # file. The following values are typically found in the Type
        # field:
        # 0000h Old
        # 0001h Standard
        # 0002h Byte-encoded
        # 0003h RGB format
        # 0004h TIFF format
        # 0005h IFF format
        # FFFFh Experimental

        # Old and standard are the same, except for the length tag.
        # byte-encoded is run-length-encoded
        # RGB looks similar to standard, but RGB byte order
        # TIFF and IFF mean that they were converted from T/IFF
        # Experimental means that it's something else.
        # (https://www.fileformat.info/format/sunraster/egff.htm)

        if file_type in (0, 1, 3, 4, 5):
            self.tile = [("raw", (0, 0) + self.size, offset, (rawmode, stride))]
        elif file_type == 2:
            self.tile = [("sun_rle", (0, 0) + self.size, offset, rawmode)]
        else:
            msg = "Unsupported Sun Raster file type"
            raise SyntaxError(msg)


#
# registry


Image.register_open(SunImageFile.format, SunImageFile, _accept)

Image.register_extension(SunImageFile.format, ".ras")
