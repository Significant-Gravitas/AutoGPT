#
# The Python Imaging Library.
# $Id$
#
# JPEG (JFIF) file handling
#
# See "Digital Compression and Coding of Continuous-Tone Still Images,
# Part 1, Requirements and Guidelines" (CCITT T.81 / ISO 10918-1)
#
# History:
# 1995-09-09 fl   Created
# 1995-09-13 fl   Added full parser
# 1996-03-25 fl   Added hack to use the IJG command line utilities
# 1996-05-05 fl   Workaround Photoshop 2.5 CMYK polarity bug
# 1996-05-28 fl   Added draft support, JFIF version (0.1)
# 1996-12-30 fl   Added encoder options, added progression property (0.2)
# 1997-08-27 fl   Save mode 1 images as BW (0.3)
# 1998-07-12 fl   Added YCbCr to draft and save methods (0.4)
# 1998-10-19 fl   Don't hang on files using 16-bit DQT's (0.4.1)
# 2001-04-16 fl   Extract DPI settings from JFIF files (0.4.2)
# 2002-07-01 fl   Skip pad bytes before markers; identify Exif files (0.4.3)
# 2003-04-25 fl   Added experimental EXIF decoder (0.5)
# 2003-06-06 fl   Added experimental EXIF GPSinfo decoder
# 2003-09-13 fl   Extract COM markers
# 2009-09-06 fl   Added icc_profile support (from Florian Hoech)
# 2009-03-06 fl   Changed CMYK handling; always use Adobe polarity (0.6)
# 2009-03-08 fl   Added subsampling support (from Justin Huff).
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-1996 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings

from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._deprecate import deprecate
from .JpegPresets import presets

#
# Parser


def Skip(self, marker):
    n = i16(self.fp.read(2)) - 2
    ImageFile._safe_read(self.fp, n)


def APP(self, marker):
    #
    # Application marker.  Store these in the APP dictionary.
    # Also look for well-known application markers.

    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)

    app = "APP%d" % (marker & 15)

    self.app[app] = s  # compatibility
    self.applist.append((app, s))

    if marker == 0xFFE0 and s[:4] == b"JFIF":
        # extract JFIF information
        self.info["jfif"] = version = i16(s, 5)  # version
        self.info["jfif_version"] = divmod(version, 256)
        # extract JFIF properties
        try:
            jfif_unit = s[7]
            jfif_density = i16(s, 8), i16(s, 10)
        except Exception:
            pass
        else:
            if jfif_unit == 1:
                self.info["dpi"] = jfif_density
            self.info["jfif_unit"] = jfif_unit
            self.info["jfif_density"] = jfif_density
    elif marker == 0xFFE1 and s[:5] == b"Exif\0":
        if "exif" not in self.info:
            # extract EXIF information (incomplete)
            self.info["exif"] = s  # FIXME: value will change
            self._exif_offset = self.fp.tell() - n + 6
    elif marker == 0xFFE2 and s[:5] == b"FPXR\0":
        # extract FlashPix information (incomplete)
        self.info["flashpix"] = s  # FIXME: value will change
    elif marker == 0xFFE2 and s[:12] == b"ICC_PROFILE\0":
        # Since an ICC profile can be larger than the maximum size of
        # a JPEG marker (64K), we need provisions to split it into
        # multiple markers. The format defined by the ICC specifies
        # one or more APP2 markers containing the following data:
        #   Identifying string      ASCII "ICC_PROFILE\0"  (12 bytes)
        #   Marker sequence number  1, 2, etc (1 byte)
        #   Number of markers       Total of APP2's used (1 byte)
        #   Profile data            (remainder of APP2 data)
        # Decoders should use the marker sequence numbers to
        # reassemble the profile, rather than assuming that the APP2
        # markers appear in the correct sequence.
        self.icclist.append(s)
    elif marker == 0xFFED and s[:14] == b"Photoshop 3.0\x00":
        # parse the image resource block
        offset = 14
        photoshop = self.info.setdefault("photoshop", {})
        while s[offset : offset + 4] == b"8BIM":
            try:
                offset += 4
                # resource code
                code = i16(s, offset)
                offset += 2
                # resource name (usually empty)
                name_len = s[offset]
                # name = s[offset+1:offset+1+name_len]
                offset += 1 + name_len
                offset += offset & 1  # align
                # resource data block
                size = i32(s, offset)
                offset += 4
                data = s[offset : offset + size]
                if code == 0x03ED:  # ResolutionInfo
                    data = {
                        "XResolution": i32(data, 0) / 65536,
                        "DisplayedUnitsX": i16(data, 4),
                        "YResolution": i32(data, 8) / 65536,
                        "DisplayedUnitsY": i16(data, 12),
                    }
                photoshop[code] = data
                offset += size
                offset += offset & 1  # align
            except struct.error:
                break  # insufficient data

    elif marker == 0xFFEE and s[:5] == b"Adobe":
        self.info["adobe"] = i16(s, 5)
        # extract Adobe custom properties
        try:
            adobe_transform = s[11]
        except IndexError:
            pass
        else:
            self.info["adobe_transform"] = adobe_transform
    elif marker == 0xFFE2 and s[:4] == b"MPF\0":
        # extract MPO information
        self.info["mp"] = s[4:]
        # offset is current location minus buffer size
        # plus constant header size
        self.info["mpoffset"] = self.fp.tell() - n + 4

    # If DPI isn't in JPEG header, fetch from EXIF
    if "dpi" not in self.info and "exif" in self.info:
        try:
            exif = self.getexif()
            resolution_unit = exif[0x0128]
            x_resolution = exif[0x011A]
            try:
                dpi = float(x_resolution[0]) / x_resolution[1]
            except TypeError:
                dpi = x_resolution
            if math.isnan(dpi):
                raise ValueError
            if resolution_unit == 3:  # cm
                # 1 dpcm = 2.54 dpi
                dpi *= 2.54
            self.info["dpi"] = dpi, dpi
        except (TypeError, KeyError, SyntaxError, ValueError, ZeroDivisionError):
            # SyntaxError for invalid/unreadable EXIF
            # KeyError for dpi not included
            # ZeroDivisionError for invalid dpi rational value
            # ValueError or TypeError for dpi being an invalid float
            self.info["dpi"] = 72, 72


def COM(self, marker):
    #
    # Comment marker.  Store these in the APP dictionary.
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)

    self.info["comment"] = s
    self.app["COM"] = s  # compatibility
    self.applist.append(("COM", s))


def SOF(self, marker):
    #
    # Start of frame marker.  Defines the size and mode of the
    # image.  JPEG is colour blind, so we use some simple
    # heuristics to map the number of layers to an appropriate
    # mode.  Note that this could be made a bit brighter, by
    # looking for JFIF and Adobe APP markers.

    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    self._size = i16(s, 3), i16(s, 1)

    self.bits = s[0]
    if self.bits != 8:
        msg = f"cannot handle {self.bits}-bit layers"
        raise SyntaxError(msg)

    self.layers = s[5]
    if self.layers == 1:
        self.mode = "L"
    elif self.layers == 3:
        self.mode = "RGB"
    elif self.layers == 4:
        self.mode = "CMYK"
    else:
        msg = f"cannot handle {self.layers}-layer images"
        raise SyntaxError(msg)

    if marker in [0xFFC2, 0xFFC6, 0xFFCA, 0xFFCE]:
        self.info["progressive"] = self.info["progression"] = 1

    if self.icclist:
        # fixup icc profile
        self.icclist.sort()  # sort by sequence number
        if self.icclist[0][13] == len(self.icclist):
            profile = []
            for p in self.icclist:
                profile.append(p[14:])
            icc_profile = b"".join(profile)
        else:
            icc_profile = None  # wrong number of fragments
        self.info["icc_profile"] = icc_profile
        self.icclist = []

    for i in range(6, len(s), 3):
        t = s[i : i + 3]
        # 4-tuples: id, vsamp, hsamp, qtable
        self.layer.append((t[0], t[1] // 16, t[1] & 15, t[2]))


def DQT(self, marker):
    #
    # Define quantization table.  Note that there might be more
    # than one table in each marker.

    # FIXME: The quantization tables can be used to estimate the
    # compression quality.

    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    while len(s):
        v = s[0]
        precision = 1 if (v // 16 == 0) else 2  # in bytes
        qt_length = 1 + precision * 64
        if len(s) < qt_length:
            msg = "bad quantization table marker"
            raise SyntaxError(msg)
        data = array.array("B" if precision == 1 else "H", s[1:qt_length])
        if sys.byteorder == "little" and precision > 1:
            data.byteswap()  # the values are always big-endian
        self.quantization[v & 15] = [data[i] for i in zigzag_index]
        s = s[qt_length:]


#
# JPEG marker table

MARKER = {
    0xFFC0: ("SOF0", "Baseline DCT", SOF),
    0xFFC1: ("SOF1", "Extended Sequential DCT", SOF),
    0xFFC2: ("SOF2", "Progressive DCT", SOF),
    0xFFC3: ("SOF3", "Spatial lossless", SOF),
    0xFFC4: ("DHT", "Define Huffman table", Skip),
    0xFFC5: ("SOF5", "Differential sequential DCT", SOF),
    0xFFC6: ("SOF6", "Differential progressive DCT", SOF),
    0xFFC7: ("SOF7", "Differential spatial", SOF),
    0xFFC8: ("JPG", "Extension", None),
    0xFFC9: ("SOF9", "Extended sequential DCT (AC)", SOF),
    0xFFCA: ("SOF10", "Progressive DCT (AC)", SOF),
    0xFFCB: ("SOF11", "Spatial lossless DCT (AC)", SOF),
    0xFFCC: ("DAC", "Define arithmetic coding conditioning", Skip),
    0xFFCD: ("SOF13", "Differential sequential DCT (AC)", SOF),
    0xFFCE: ("SOF14", "Differential progressive DCT (AC)", SOF),
    0xFFCF: ("SOF15", "Differential spatial (AC)", SOF),
    0xFFD0: ("RST0", "Restart 0", None),
    0xFFD1: ("RST1", "Restart 1", None),
    0xFFD2: ("RST2", "Restart 2", None),
    0xFFD3: ("RST3", "Restart 3", None),
    0xFFD4: ("RST4", "Restart 4", None),
    0xFFD5: ("RST5", "Restart 5", None),
    0xFFD6: ("RST6", "Restart 6", None),
    0xFFD7: ("RST7", "Restart 7", None),
    0xFFD8: ("SOI", "Start of image", None),
    0xFFD9: ("EOI", "End of image", None),
    0xFFDA: ("SOS", "Start of scan", Skip),
    0xFFDB: ("DQT", "Define quantization table", DQT),
    0xFFDC: ("DNL", "Define number of lines", Skip),
    0xFFDD: ("DRI", "Define restart interval", Skip),
    0xFFDE: ("DHP", "Define hierarchical progression", SOF),
    0xFFDF: ("EXP", "Expand reference component", Skip),
    0xFFE0: ("APP0", "Application segment 0", APP),
    0xFFE1: ("APP1", "Application segment 1", APP),
    0xFFE2: ("APP2", "Application segment 2", APP),
    0xFFE3: ("APP3", "Application segment 3", APP),
    0xFFE4: ("APP4", "Application segment 4", APP),
    0xFFE5: ("APP5", "Application segment 5", APP),
    0xFFE6: ("APP6", "Application segment 6", APP),
    0xFFE7: ("APP7", "Application segment 7", APP),
    0xFFE8: ("APP8", "Application segment 8", APP),
    0xFFE9: ("APP9", "Application segment 9", APP),
    0xFFEA: ("APP10", "Application segment 10", APP),
    0xFFEB: ("APP11", "Application segment 11", APP),
    0xFFEC: ("APP12", "Application segment 12", APP),
    0xFFED: ("APP13", "Application segment 13", APP),
    0xFFEE: ("APP14", "Application segment 14", APP),
    0xFFEF: ("APP15", "Application segment 15", APP),
    0xFFF0: ("JPG0", "Extension 0", None),
    0xFFF1: ("JPG1", "Extension 1", None),
    0xFFF2: ("JPG2", "Extension 2", None),
    0xFFF3: ("JPG3", "Extension 3", None),
    0xFFF4: ("JPG4", "Extension 4", None),
    0xFFF5: ("JPG5", "Extension 5", None),
    0xFFF6: ("JPG6", "Extension 6", None),
    0xFFF7: ("JPG7", "Extension 7", None),
    0xFFF8: ("JPG8", "Extension 8", None),
    0xFFF9: ("JPG9", "Extension 9", None),
    0xFFFA: ("JPG10", "Extension 10", None),
    0xFFFB: ("JPG11", "Extension 11", None),
    0xFFFC: ("JPG12", "Extension 12", None),
    0xFFFD: ("JPG13", "Extension 13", None),
    0xFFFE: ("COM", "Comment", COM),
}


def _accept(prefix):
    # Magic number was taken from https://en.wikipedia.org/wiki/JPEG
    return prefix[:3] == b"\xFF\xD8\xFF"


##
# Image plugin for JPEG and JFIF images.


class JpegImageFile(ImageFile.ImageFile):
    format = "JPEG"
    format_description = "JPEG (ISO 10918)"

    def _open(self):
        s = self.fp.read(3)

        if not _accept(s):
            msg = "not a JPEG file"
            raise SyntaxError(msg)
        s = b"\xFF"

        # Create attributes
        self.bits = self.layers = 0

        # JPEG specifics (internal)
        self.layer = []
        self.huffman_dc = {}
        self.huffman_ac = {}
        self.quantization = {}
        self.app = {}  # compatibility
        self.applist = []
        self.icclist = []

        while True:
            i = s[0]
            if i == 0xFF:
                s = s + self.fp.read(1)
                i = i16(s)
            else:
                # Skip non-0xFF junk
                s = self.fp.read(1)
                continue

            if i in MARKER:
                name, description, handler = MARKER[i]
                if handler is not None:
                    handler(self, i)
                if i == 0xFFDA:  # start of scan
                    rawmode = self.mode
                    if self.mode == "CMYK":
                        rawmode = "CMYK;I"  # assume adobe conventions
                    self.tile = [("jpeg", (0, 0) + self.size, 0, (rawmode, ""))]
                    # self.__offset = self.fp.tell()
                    break
                s = self.fp.read(1)
            elif i == 0 or i == 0xFFFF:
                # padded marker or junk; move on
                s = b"\xff"
            elif i == 0xFF00:  # Skip extraneous data (escaped 0xFF)
                s = self.fp.read(1)
            else:
                msg = "no marker found"
                raise SyntaxError(msg)

    def load_read(self, read_bytes):
        """
        internal: read more image data
        For premature EOF and LOAD_TRUNCATED_IMAGES adds EOI marker
        so libjpeg can finish decoding
        """
        s = self.fp.read(read_bytes)

        if not s and ImageFile.LOAD_TRUNCATED_IMAGES and not hasattr(self, "_ended"):
            # Premature EOF.
            # Pretend file is finished adding EOI marker
            self._ended = True
            return b"\xFF\xD9"

        return s

    def draft(self, mode, size):
        if len(self.tile) != 1:
            return

        # Protect from second call
        if self.decoderconfig:
            return

        d, e, o, a = self.tile[0]
        scale = 1
        original_size = self.size

        if a[0] == "RGB" and mode in ["L", "YCbCr"]:
            self.mode = mode
            a = mode, ""

        if size:
            scale = min(self.size[0] // size[0], self.size[1] // size[1])
            for s in [8, 4, 2, 1]:
                if scale >= s:
                    break
            e = (
                e[0],
                e[1],
                (e[2] - e[0] + s - 1) // s + e[0],
                (e[3] - e[1] + s - 1) // s + e[1],
            )
            self._size = ((self.size[0] + s - 1) // s, (self.size[1] + s - 1) // s)
            scale = s

        self.tile = [(d, e, o, a)]
        self.decoderconfig = (scale, 0)

        box = (0, 0, original_size[0] / scale, original_size[1] / scale)
        return self.mode, box

    def load_djpeg(self):
        # ALTERNATIVE: handle JPEGs via the IJG command line utilities

        f, path = tempfile.mkstemp()
        os.close(f)
        if os.path.exists(self.filename):
            subprocess.check_call(["djpeg", "-outfile", path, self.filename])
        else:
            msg = "Invalid Filename"
            raise ValueError(msg)

        try:
            with Image.open(path) as _im:
                _im.load()
                self.im = _im.im
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        self.mode = self.im.mode
        self._size = self.im.size

        self.tile = []

    def _getexif(self):
        return _getexif(self)

    def _getmp(self):
        return _getmp(self)

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """

        for segment, content in self.applist:
            if segment == "APP1":
                marker, xmp_tags = content.rsplit(b"\x00", 1)
                if marker == b"http://ns.adobe.com/xap/1.0/":
                    return self._getxmp(xmp_tags)
        return {}


def _getexif(self):
    if "exif" not in self.info:
        return None
    return self.getexif()._get_merged_dict()


def _getmp(self):
    # Extract MP information.  This method was inspired by the "highly
    # experimental" _getexif version that's been in use for years now,
    # itself based on the ImageFileDirectory class in the TIFF plugin.

    # The MP record essentially consists of a TIFF file embedded in a JPEG
    # application marker.
    try:
        data = self.info["mp"]
    except KeyError:
        return None
    file_contents = io.BytesIO(data)
    head = file_contents.read(8)
    endianness = ">" if head[:4] == b"\x4d\x4d\x00\x2a" else "<"
    # process dictionary
    from . import TiffImagePlugin

    try:
        info = TiffImagePlugin.ImageFileDirectory_v2(head)
        file_contents.seek(info.next)
        info.load(file_contents)
        mp = dict(info)
    except Exception as e:
        msg = "malformed MP Index (unreadable directory)"
        raise SyntaxError(msg) from e
    # it's an error not to have a number of images
    try:
        quant = mp[0xB001]
    except KeyError as e:
        msg = "malformed MP Index (no number of images)"
        raise SyntaxError(msg) from e
    # get MP entries
    mpentries = []
    try:
        rawmpentries = mp[0xB002]
        for entrynum in range(0, quant):
            unpackedentry = struct.unpack_from(
                f"{endianness}LLLHH", rawmpentries, entrynum * 16
            )
            labels = ("Attribute", "Size", "DataOffset", "EntryNo1", "EntryNo2")
            mpentry = dict(zip(labels, unpackedentry))
            mpentryattr = {
                "DependentParentImageFlag": bool(mpentry["Attribute"] & (1 << 31)),
                "DependentChildImageFlag": bool(mpentry["Attribute"] & (1 << 30)),
                "RepresentativeImageFlag": bool(mpentry["Attribute"] & (1 << 29)),
                "Reserved": (mpentry["Attribute"] & (3 << 27)) >> 27,
                "ImageDataFormat": (mpentry["Attribute"] & (7 << 24)) >> 24,
                "MPType": mpentry["Attribute"] & 0x00FFFFFF,
            }
            if mpentryattr["ImageDataFormat"] == 0:
                mpentryattr["ImageDataFormat"] = "JPEG"
            else:
                msg = "unsupported picture format in MPO"
                raise SyntaxError(msg)
            mptypemap = {
                0x000000: "Undefined",
                0x010001: "Large Thumbnail (VGA Equivalent)",
                0x010002: "Large Thumbnail (Full HD Equivalent)",
                0x020001: "Multi-Frame Image (Panorama)",
                0x020002: "Multi-Frame Image: (Disparity)",
                0x020003: "Multi-Frame Image: (Multi-Angle)",
                0x030000: "Baseline MP Primary Image",
            }
            mpentryattr["MPType"] = mptypemap.get(mpentryattr["MPType"], "Unknown")
            mpentry["Attribute"] = mpentryattr
            mpentries.append(mpentry)
        mp[0xB002] = mpentries
    except KeyError as e:
        msg = "malformed MP Index (bad MP Entry)"
        raise SyntaxError(msg) from e
    # Next we should try and parse the individual image unique ID list;
    # we don't because I've never seen this actually used in a real MPO
    # file and so can't test it.
    return mp


# --------------------------------------------------------------------
# stuff to save JPEG files

RAWMODE = {
    "1": "L",
    "L": "L",
    "RGB": "RGB",
    "RGBX": "RGB",
    "CMYK": "CMYK;I",  # assume adobe conventions
    "YCbCr": "YCbCr",
}

# fmt: off
zigzag_index = (
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
)

samplings = {
    (1, 1, 1, 1, 1, 1): 0,
    (2, 1, 1, 1, 1, 1): 1,
    (2, 2, 1, 1, 1, 1): 2,
}
# fmt: on


def convert_dict_qtables(qtables):
    deprecate("convert_dict_qtables", 10, action="Conversion is no longer needed")
    return qtables


def get_sampling(im):
    # There's no subsampling when images have only 1 layer
    # (grayscale images) or when they are CMYK (4 layers),
    # so set subsampling to the default value.
    #
    # NOTE: currently Pillow can't encode JPEG to YCCK format.
    # If YCCK support is added in the future, subsampling code will have
    # to be updated (here and in JpegEncode.c) to deal with 4 layers.
    if not hasattr(im, "layers") or im.layers in (1, 4):
        return -1
    sampling = im.layer[0][1:3] + im.layer[1][1:3] + im.layer[2][1:3]
    return samplings.get(sampling, -1)


def _save(im, fp, filename):
    if im.width == 0 or im.height == 0:
        msg = "cannot write empty image as JPEG"
        raise ValueError(msg)

    try:
        rawmode = RAWMODE[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as JPEG"
        raise OSError(msg) from e

    info = im.encoderinfo

    dpi = [round(x) for x in info.get("dpi", (0, 0))]

    quality = info.get("quality", -1)
    subsampling = info.get("subsampling", -1)
    qtables = info.get("qtables")

    if quality == "keep":
        quality = -1
        subsampling = "keep"
        qtables = "keep"
    elif quality in presets:
        preset = presets[quality]
        quality = -1
        subsampling = preset.get("subsampling", -1)
        qtables = preset.get("quantization")
    elif not isinstance(quality, int):
        msg = "Invalid quality setting"
        raise ValueError(msg)
    else:
        if subsampling in presets:
            subsampling = presets[subsampling].get("subsampling", -1)
        if isinstance(qtables, str) and qtables in presets:
            qtables = presets[qtables].get("quantization")

    if subsampling == "4:4:4":
        subsampling = 0
    elif subsampling == "4:2:2":
        subsampling = 1
    elif subsampling == "4:2:0":
        subsampling = 2
    elif subsampling == "4:1:1":
        # For compatibility. Before Pillow 4.3, 4:1:1 actually meant 4:2:0.
        # Set 4:2:0 if someone is still using that value.
        subsampling = 2
    elif subsampling == "keep":
        if im.format != "JPEG":
            msg = "Cannot use 'keep' when original image is not a JPEG"
            raise ValueError(msg)
        subsampling = get_sampling(im)

    def validate_qtables(qtables):
        if qtables is None:
            return qtables
        if isinstance(qtables, str):
            try:
                lines = [
                    int(num)
                    for line in qtables.splitlines()
                    for num in line.split("#", 1)[0].split()
                ]
            except ValueError as e:
                msg = "Invalid quantization table"
                raise ValueError(msg) from e
            else:
                qtables = [lines[s : s + 64] for s in range(0, len(lines), 64)]
        if isinstance(qtables, (tuple, list, dict)):
            if isinstance(qtables, dict):
                qtables = [
                    qtables[key] for key in range(len(qtables)) if key in qtables
                ]
            elif isinstance(qtables, tuple):
                qtables = list(qtables)
            if not (0 < len(qtables) < 5):
                msg = "None or too many quantization tables"
                raise ValueError(msg)
            for idx, table in enumerate(qtables):
                try:
                    if len(table) != 64:
                        raise TypeError
                    table = array.array("H", table)
                except TypeError as e:
                    msg = "Invalid quantization table"
                    raise ValueError(msg) from e
                else:
                    qtables[idx] = list(table)
            return qtables

    if qtables == "keep":
        if im.format != "JPEG":
            msg = "Cannot use 'keep' when original image is not a JPEG"
            raise ValueError(msg)
        qtables = getattr(im, "quantization", None)
    qtables = validate_qtables(qtables)

    extra = info.get("extra", b"")

    MAX_BYTES_IN_MARKER = 65533
    icc_profile = info.get("icc_profile")
    if icc_profile:
        ICC_OVERHEAD_LEN = 14
        MAX_DATA_BYTES_IN_MARKER = MAX_BYTES_IN_MARKER - ICC_OVERHEAD_LEN
        markers = []
        while icc_profile:
            markers.append(icc_profile[:MAX_DATA_BYTES_IN_MARKER])
            icc_profile = icc_profile[MAX_DATA_BYTES_IN_MARKER:]
        i = 1
        for marker in markers:
            size = o16(2 + ICC_OVERHEAD_LEN + len(marker))
            extra += (
                b"\xFF\xE2"
                + size
                + b"ICC_PROFILE\0"
                + o8(i)
                + o8(len(markers))
                + marker
            )
            i += 1

    comment = info.get("comment", im.info.get("comment"))

    # "progressive" is the official name, but older documentation
    # says "progression"
    # FIXME: issue a warning if the wrong form is used (post-1.1.7)
    progressive = info.get("progressive", False) or info.get("progression", False)

    optimize = info.get("optimize", False)

    exif = info.get("exif", b"")
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    if len(exif) > MAX_BYTES_IN_MARKER:
        msg = "EXIF data is too long"
        raise ValueError(msg)

    # get keyword arguments
    im.encoderconfig = (
        quality,
        progressive,
        info.get("smooth", 0),
        optimize,
        info.get("streamtype", 0),
        dpi[0],
        dpi[1],
        subsampling,
        qtables,
        comment,
        extra,
        exif,
    )

    # if we optimize, libjpeg needs a buffer big enough to hold the whole image
    # in a shot. Guessing on the size, at im.size bytes. (raw pixel size is
    # channels*size, this is a value that's been used in a django patch.
    # https://github.com/matthewwithanm/django-imagekit/issues/50
    bufsize = 0
    if optimize or progressive:
        # CMYK can be bigger
        if im.mode == "CMYK":
            bufsize = 4 * im.size[0] * im.size[1]
        # keep sets quality to -1, but the actual value may be high.
        elif quality >= 95 or quality == -1:
            bufsize = 2 * im.size[0] * im.size[1]
        else:
            bufsize = im.size[0] * im.size[1]

    # The EXIF info needs to be written as one block, + APP1, + one spare byte.
    # Ensure that our buffer is big enough. Same with the icc_profile block.
    bufsize = max(ImageFile.MAXBLOCK, bufsize, len(exif) + 5, len(extra) + 1)

    ImageFile._save(im, fp, [("jpeg", (0, 0) + im.size, 0, rawmode)], bufsize)


def _save_cjpeg(im, fp, filename):
    # ALTERNATIVE: handle JPEGs via the IJG command line utilities.
    tempfile = im._dump()
    subprocess.check_call(["cjpeg", "-outfile", filename, tempfile])
    try:
        os.unlink(tempfile)
    except OSError:
        pass


##
# Factory for making JPEG and MPO instances
def jpeg_factory(fp=None, filename=None):
    im = JpegImageFile(fp, filename)
    try:
        mpheader = im._getmp()
        if mpheader[45057] > 1:
            # It's actually an MPO
            from .MpoImagePlugin import MpoImageFile

            # Don't reload everything, just convert it.
            im = MpoImageFile.adopt(im, mpheader)
    except (TypeError, IndexError):
        # It is really a JPEG
        pass
    except SyntaxError:
        warnings.warn(
            "Image appears to be a malformed MPO file, it will be "
            "interpreted as a base JPEG file"
        )
    return im


# ---------------------------------------------------------------------
# Registry stuff

Image.register_open(JpegImageFile.format, jpeg_factory, _accept)
Image.register_save(JpegImageFile.format, _save)

Image.register_extensions(JpegImageFile.format, [".jfif", ".jpe", ".jpg", ".jpeg"])

Image.register_mime(JpegImageFile.format, "image/jpeg")
