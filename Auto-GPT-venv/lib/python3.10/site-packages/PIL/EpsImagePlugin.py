#
# The Python Imaging Library.
# $Id$
#
# EPS file handling
#
# History:
# 1995-09-01 fl   Created (0.1)
# 1996-05-18 fl   Don't choke on "atend" fields, Ghostscript interface (0.2)
# 1996-08-22 fl   Don't choke on floating point BoundingBox values
# 1996-08-23 fl   Handle files from Macintosh (0.3)
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.4)
# 2003-09-07 fl   Check gs.close status (from Federico Di Gregorio) (0.5)
# 2014-05-07 e    Handling of EPS with binary preview and fixed resolution
#                 resizing
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import io
import os
import re
import subprocess
import sys
import tempfile

from . import Image, ImageFile
from ._binary import i32le as i32
from ._deprecate import deprecate

# --------------------------------------------------------------------


split = re.compile(r"^%%([^:]*):[ \t]*(.*)[ \t]*$")
field = re.compile(r"^%[%!\w]([^:]*)[ \t]*$")

gs_windows_binary = None
if sys.platform.startswith("win"):
    import shutil

    for binary in ("gswin32c", "gswin64c", "gs"):
        if shutil.which(binary) is not None:
            gs_windows_binary = binary
            break
    else:
        gs_windows_binary = False


def has_ghostscript():
    if gs_windows_binary:
        return True
    if not sys.platform.startswith("win"):
        try:
            subprocess.check_call(["gs", "--version"], stdout=subprocess.DEVNULL)
            return True
        except OSError:
            # No Ghostscript
            pass
    return False


def Ghostscript(tile, size, fp, scale=1, transparency=False):
    """Render an image using Ghostscript"""

    # Unpack decoder tile
    decoder, tile, offset, data = tile[0]
    length, bbox = data

    # Hack to support hi-res rendering
    scale = int(scale) or 1
    # orig_size = size
    # orig_bbox = bbox
    size = (size[0] * scale, size[1] * scale)
    # resolution is dependent on bbox and size
    res = (
        72.0 * size[0] / (bbox[2] - bbox[0]),
        72.0 * size[1] / (bbox[3] - bbox[1]),
    )

    out_fd, outfile = tempfile.mkstemp()
    os.close(out_fd)

    infile_temp = None
    if hasattr(fp, "name") and os.path.exists(fp.name):
        infile = fp.name
    else:
        in_fd, infile_temp = tempfile.mkstemp()
        os.close(in_fd)
        infile = infile_temp

        # Ignore length and offset!
        # Ghostscript can read it
        # Copy whole file to read in Ghostscript
        with open(infile_temp, "wb") as f:
            # fetch length of fp
            fp.seek(0, io.SEEK_END)
            fsize = fp.tell()
            # ensure start position
            # go back
            fp.seek(0)
            lengthfile = fsize
            while lengthfile > 0:
                s = fp.read(min(lengthfile, 100 * 1024))
                if not s:
                    break
                lengthfile -= len(s)
                f.write(s)

    device = "pngalpha" if transparency else "ppmraw"

    # Build Ghostscript command
    command = [
        "gs",
        "-q",  # quiet mode
        "-g%dx%d" % size,  # set output geometry (pixels)
        "-r%fx%f" % res,  # set input DPI (dots per inch)
        "-dBATCH",  # exit after processing
        "-dNOPAUSE",  # don't pause between pages
        "-dSAFER",  # safe mode
        f"-sDEVICE={device}",
        f"-sOutputFile={outfile}",  # output file
        # adjust for image origin
        "-c",
        f"{-bbox[0]} {-bbox[1]} translate",
        "-f",
        infile,  # input file
        # showpage (see https://bugs.ghostscript.com/show_bug.cgi?id=698272)
        "-c",
        "showpage",
    ]

    if gs_windows_binary is not None:
        if not gs_windows_binary:
            msg = "Unable to locate Ghostscript on paths"
            raise OSError(msg)
        command[0] = gs_windows_binary

    # push data through Ghostscript
    try:
        startupinfo = None
        if sys.platform.startswith("win"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.check_call(command, startupinfo=startupinfo)
        out_im = Image.open(outfile)
        out_im.load()
    finally:
        try:
            os.unlink(outfile)
            if infile_temp:
                os.unlink(infile_temp)
        except OSError:
            pass

    im = out_im.im.copy()
    out_im.close()
    return im


class PSFile:
    """
    Wrapper for bytesio object that treats either CR or LF as end of line.
    This class is no longer used internally, but kept for backwards compatibility.
    """

    def __init__(self, fp):
        deprecate(
            "PSFile",
            11,
            action="If you need the functionality of this class "
            "you will need to implement it yourself.",
        )
        self.fp = fp
        self.char = None

    def seek(self, offset, whence=io.SEEK_SET):
        self.char = None
        self.fp.seek(offset, whence)

    def readline(self):
        s = [self.char or b""]
        self.char = None

        c = self.fp.read(1)
        while (c not in b"\r\n") and len(c):
            s.append(c)
            c = self.fp.read(1)

        self.char = self.fp.read(1)
        # line endings can be 1 or 2 of \r \n, in either order
        if self.char in b"\r\n":
            self.char = None

        return b"".join(s).decode("latin-1")


def _accept(prefix):
    return prefix[:4] == b"%!PS" or (len(prefix) >= 4 and i32(prefix) == 0xC6D3D0C5)


##
# Image plugin for Encapsulated PostScript. This plugin supports only
# a few variants of this format.


class EpsImageFile(ImageFile.ImageFile):
    """EPS File Parser for the Python Imaging Library"""

    format = "EPS"
    format_description = "Encapsulated Postscript"

    mode_map = {1: "L", 2: "LAB", 3: "RGB", 4: "CMYK"}

    def _open(self):
        (length, offset) = self._find_offset(self.fp)

        # go to offset - start of "%!PS"
        self.fp.seek(offset)

        self.mode = "RGB"
        self._size = None

        byte_arr = bytearray(255)
        bytes_mv = memoryview(byte_arr)
        bytes_read = 0
        reading_comments = True

        def check_required_header_comments():
            if "PS-Adobe" not in self.info:
                msg = 'EPS header missing "%!PS-Adobe" comment'
                raise SyntaxError(msg)
            if "BoundingBox" not in self.info:
                msg = 'EPS header missing "%%BoundingBox" comment'
                raise SyntaxError(msg)

        while True:
            byte = self.fp.read(1)
            if byte == b"":
                # if we didn't read a byte we must be at the end of the file
                if bytes_read == 0:
                    break
            elif byte in b"\r\n":
                # if we read a line ending character, ignore it and parse what
                # we have already read. if we haven't read any other characters,
                # continue reading
                if bytes_read == 0:
                    continue
            else:
                # ASCII/hexadecimal lines in an EPS file must not exceed
                # 255 characters, not including line ending characters
                if bytes_read >= 255:
                    # only enforce this for lines starting with a "%",
                    # otherwise assume it's binary data
                    if byte_arr[0] == ord("%"):
                        msg = "not an EPS file"
                        raise SyntaxError(msg)
                    else:
                        if reading_comments:
                            check_required_header_comments()
                            reading_comments = False
                        # reset bytes_read so we can keep reading
                        # data until the end of the line
                        bytes_read = 0
                byte_arr[bytes_read] = byte[0]
                bytes_read += 1
                continue

            if reading_comments:
                # Load EPS header

                # if this line doesn't start with a "%",
                # or does start with "%%EndComments",
                # then we've reached the end of the header/comments
                if byte_arr[0] != ord("%") or bytes_mv[:13] == b"%%EndComments":
                    check_required_header_comments()
                    reading_comments = False
                    continue

                s = str(bytes_mv[:bytes_read], "latin-1")

                try:
                    m = split.match(s)
                except re.error as e:
                    msg = "not an EPS file"
                    raise SyntaxError(msg) from e

                if m:
                    k, v = m.group(1, 2)
                    self.info[k] = v
                    if k == "BoundingBox":
                        try:
                            # Note: The DSC spec says that BoundingBox
                            # fields should be integers, but some drivers
                            # put floating point values there anyway.
                            box = [int(float(i)) for i in v.split()]
                            self._size = box[2] - box[0], box[3] - box[1]
                            self.tile = [
                                ("eps", (0, 0) + self.size, offset, (length, box))
                            ]
                        except Exception:
                            pass
                else:
                    m = field.match(s)
                    if m:
                        k = m.group(1)
                        if k[:8] == "PS-Adobe":
                            self.info["PS-Adobe"] = k[9:]
                        else:
                            self.info[k] = ""
                    elif s[0] == "%":
                        # handle non-DSC PostScript comments that some
                        # tools mistakenly put in the Comments section
                        pass
                    else:
                        msg = "bad EPS header"
                        raise OSError(msg)
            elif bytes_mv[:11] == b"%ImageData:":
                # Check for an "ImageData" descriptor
                # https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577413_pgfId-1035096

                # Values:
                # columns
                # rows
                # bit depth (1 or 8)
                # mode (1: L, 2: LAB, 3: RGB, 4: CMYK)
                # number of padding channels
                # block size (number of bytes per row per channel)
                # binary/ascii (1: binary, 2: ascii)
                # data start identifier (the image data follows after a single line
                #   consisting only of this quoted value)
                image_data_values = byte_arr[11:bytes_read].split(None, 7)
                columns, rows, bit_depth, mode_id = [
                    int(value) for value in image_data_values[:4]
                ]

                if bit_depth == 1:
                    self.mode = "1"
                elif bit_depth == 8:
                    try:
                        self.mode = self.mode_map[mode_id]
                    except ValueError:
                        break
                else:
                    break

                self._size = columns, rows
                return

            bytes_read = 0

        check_required_header_comments()

        if not self._size:
            self._size = 1, 1  # errors if this isn't set. why (1,1)?
            msg = "cannot determine EPS bounding box"
            raise OSError(msg)

    def _find_offset(self, fp):
        s = fp.read(4)

        if s == b"%!PS":
            # for HEAD without binary preview
            fp.seek(0, io.SEEK_END)
            length = fp.tell()
            offset = 0
        elif i32(s) == 0xC6D3D0C5:
            # FIX for: Some EPS file not handled correctly / issue #302
            # EPS can contain binary data
            # or start directly with latin coding
            # more info see:
            # https://web.archive.org/web/20160528181353/http://partners.adobe.com/public/developer/en/ps/5002.EPSF_Spec.pdf
            s = fp.read(8)
            offset = i32(s)
            length = i32(s, 4)
        else:
            msg = "not an EPS file"
            raise SyntaxError(msg)

        return length, offset

    def load(self, scale=1, transparency=False):
        # Load EPS via Ghostscript
        if self.tile:
            self.im = Ghostscript(self.tile, self.size, self.fp, scale, transparency)
            self.mode = self.im.mode
            self._size = self.im.size
            self.tile = []
        return Image.Image.load(self)

    def load_seek(self, *args, **kwargs):
        # we can't incrementally load, so force ImageFile.parser to
        # use our custom load method by defining this method.
        pass


# --------------------------------------------------------------------


def _save(im, fp, filename, eps=1):
    """EPS Writer for the Python Imaging Library."""

    # make sure image data is available
    im.load()

    # determine PostScript image mode
    if im.mode == "L":
        operator = (8, 1, b"image")
    elif im.mode == "RGB":
        operator = (8, 3, b"false 3 colorimage")
    elif im.mode == "CMYK":
        operator = (8, 4, b"false 4 colorimage")
    else:
        msg = "image mode is not supported"
        raise ValueError(msg)

    if eps:
        # write EPS header
        fp.write(b"%!PS-Adobe-3.0 EPSF-3.0\n")
        fp.write(b"%%Creator: PIL 0.1 EpsEncode\n")
        # fp.write("%%CreationDate: %s"...)
        fp.write(b"%%%%BoundingBox: 0 0 %d %d\n" % im.size)
        fp.write(b"%%Pages: 1\n")
        fp.write(b"%%EndComments\n")
        fp.write(b"%%Page: 1 1\n")
        fp.write(b"%%ImageData: %d %d " % im.size)
        fp.write(b'%d %d 0 1 1 "%s"\n' % operator)

    # image header
    fp.write(b"gsave\n")
    fp.write(b"10 dict begin\n")
    fp.write(b"/buf %d string def\n" % (im.size[0] * operator[1]))
    fp.write(b"%d %d scale\n" % im.size)
    fp.write(b"%d %d 8\n" % im.size)  # <= bits
    fp.write(b"[%d 0 0 -%d 0 %d]\n" % (im.size[0], im.size[1], im.size[1]))
    fp.write(b"{ currentfile buf readhexstring pop } bind\n")
    fp.write(operator[2] + b"\n")
    if hasattr(fp, "flush"):
        fp.flush()

    ImageFile._save(im, fp, [("eps", (0, 0) + im.size, 0, None)])

    fp.write(b"\n%%%%EndBinary\n")
    fp.write(b"grestore end\n")
    if hasattr(fp, "flush"):
        fp.flush()


# --------------------------------------------------------------------


Image.register_open(EpsImageFile.format, EpsImageFile, _accept)

Image.register_save(EpsImageFile.format, _save)

Image.register_extensions(EpsImageFile.format, [".ps", ".eps"])

Image.register_mime(EpsImageFile.format, "application/postscript")
