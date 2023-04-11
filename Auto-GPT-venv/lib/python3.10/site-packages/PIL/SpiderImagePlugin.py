#
# The Python Imaging Library.
#
# SPIDER image file handling
#
# History:
# 2004-08-02    Created BB
# 2006-03-02    added save method
# 2006-03-13    added support for stack images
#
# Copyright (c) 2004 by Health Research Inc. (HRI) RENSSELAER, NY 12144.
# Copyright (c) 2004 by William Baxter.
# Copyright (c) 2004 by Secret Labs AB.
# Copyright (c) 2004 by Fredrik Lundh.
#

##
# Image plugin for the Spider image format. This format is used
# by the SPIDER software, in processing image data from electron
# microscopy and tomography.
##

#
# SpiderImagePlugin.py
#
# The Spider image format is used by SPIDER software, in processing
# image data from electron microscopy and tomography.
#
# Spider home page:
# https://spider.wadsworth.org/spider_doc/spider/docs/spider.html
#
# Details about the Spider image format:
# https://spider.wadsworth.org/spider_doc/spider/docs/image_doc.html
#
import os
import struct
import sys

from PIL import Image, ImageFile


def isInt(f):
    try:
        i = int(f)
        if f - i == 0:
            return 1
        else:
            return 0
    except (ValueError, OverflowError):
        return 0


iforms = [1, 3, -11, -12, -21, -22]


# There is no magic number to identify Spider files, so just check a
# series of header locations to see if they have reasonable values.
# Returns no. of bytes in the header, if it is a valid Spider header,
# otherwise returns 0


def isSpiderHeader(t):
    h = (99,) + t  # add 1 value so can use spider header index start=1
    # header values 1,2,5,12,13,22,23 should be integers
    for i in [1, 2, 5, 12, 13, 22, 23]:
        if not isInt(h[i]):
            return 0
    # check iform
    iform = int(h[5])
    if iform not in iforms:
        return 0
    # check other header values
    labrec = int(h[13])  # no. records in file header
    labbyt = int(h[22])  # total no. of bytes in header
    lenbyt = int(h[23])  # record length in bytes
    if labbyt != (labrec * lenbyt):
        return 0
    # looks like a valid header
    return labbyt


def isSpiderImage(filename):
    with open(filename, "rb") as fp:
        f = fp.read(92)  # read 23 * 4 bytes
    t = struct.unpack(">23f", f)  # try big-endian first
    hdrlen = isSpiderHeader(t)
    if hdrlen == 0:
        t = struct.unpack("<23f", f)  # little-endian
        hdrlen = isSpiderHeader(t)
    return hdrlen


class SpiderImageFile(ImageFile.ImageFile):
    format = "SPIDER"
    format_description = "Spider 2D image"
    _close_exclusive_fp_after_loading = False

    def _open(self):
        # check header
        n = 27 * 4  # read 27 float values
        f = self.fp.read(n)

        try:
            self.bigendian = 1
            t = struct.unpack(">27f", f)  # try big-endian first
            hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                self.bigendian = 0
                t = struct.unpack("<27f", f)  # little-endian
                hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                msg = "not a valid Spider file"
                raise SyntaxError(msg)
        except struct.error as e:
            msg = "not a valid Spider file"
            raise SyntaxError(msg) from e

        h = (99,) + t  # add 1 value : spider header index starts at 1
        iform = int(h[5])
        if iform != 1:
            msg = "not a Spider 2D image"
            raise SyntaxError(msg)

        self._size = int(h[12]), int(h[2])  # size in pixels (width, height)
        self.istack = int(h[24])
        self.imgnumber = int(h[27])

        if self.istack == 0 and self.imgnumber == 0:
            # stk=0, img=0: a regular 2D image
            offset = hdrlen
            self._nimages = 1
        elif self.istack > 0 and self.imgnumber == 0:
            # stk>0, img=0: Opening the stack for the first time
            self.imgbytes = int(h[12]) * int(h[2]) * 4
            self.hdrlen = hdrlen
            self._nimages = int(h[26])
            # Point to the first image in the stack
            offset = hdrlen * 2
            self.imgnumber = 1
        elif self.istack == 0 and self.imgnumber > 0:
            # stk=0, img>0: an image within the stack
            offset = hdrlen + self.stkoffset
            self.istack = 2  # So Image knows it's still a stack
        else:
            msg = "inconsistent stack header values"
            raise SyntaxError(msg)

        if self.bigendian:
            self.rawmode = "F;32BF"
        else:
            self.rawmode = "F;32F"
        self.mode = "F"

        self.tile = [("raw", (0, 0) + self.size, offset, (self.rawmode, 0, 1))]
        self._fp = self.fp  # FIXME: hack

    @property
    def n_frames(self):
        return self._nimages

    @property
    def is_animated(self):
        return self._nimages > 1

    # 1st image index is zero (although SPIDER imgnumber starts at 1)
    def tell(self):
        if self.imgnumber < 1:
            return 0
        else:
            return self.imgnumber - 1

    def seek(self, frame):
        if self.istack == 0:
            msg = "attempt to seek in a non-stack file"
            raise EOFError(msg)
        if not self._seek_check(frame):
            return
        self.stkoffset = self.hdrlen + frame * (self.hdrlen + self.imgbytes)
        self.fp = self._fp
        self.fp.seek(self.stkoffset)
        self._open()

    # returns a byte image after rescaling to 0..255
    def convert2byte(self, depth=255):
        (minimum, maximum) = self.getextrema()
        m = 1
        if maximum != minimum:
            m = depth / (maximum - minimum)
        b = -m * minimum
        return self.point(lambda i, m=m, b=b: i * m + b).convert("L")

    # returns a ImageTk.PhotoImage object, after rescaling to 0..255
    def tkPhotoImage(self):
        from PIL import ImageTk

        return ImageTk.PhotoImage(self.convert2byte(), palette=256)


# --------------------------------------------------------------------
# Image series


# given a list of filenames, return a list of images
def loadImageSeries(filelist=None):
    """create a list of :py:class:`~PIL.Image.Image` objects for use in a montage"""
    if filelist is None or len(filelist) < 1:
        return

    imglist = []
    for img in filelist:
        if not os.path.exists(img):
            print(f"unable to find {img}")
            continue
        try:
            with Image.open(img) as im:
                im = im.convert2byte()
        except Exception:
            if not isSpiderImage(img):
                print(img + " is not a Spider image file")
            continue
        im.info["filename"] = img
        imglist.append(im)
    return imglist


# --------------------------------------------------------------------
# For saving images in Spider format


def makeSpiderHeader(im):
    nsam, nrow = im.size
    lenbyt = nsam * 4  # There are labrec records in the header
    labrec = int(1024 / lenbyt)
    if 1024 % lenbyt != 0:
        labrec += 1
    labbyt = labrec * lenbyt
    nvalues = int(labbyt / 4)
    if nvalues < 23:
        return []

    hdr = []
    for i in range(nvalues):
        hdr.append(0.0)

    # NB these are Fortran indices
    hdr[1] = 1.0  # nslice (=1 for an image)
    hdr[2] = float(nrow)  # number of rows per slice
    hdr[3] = float(nrow)  # number of records in the image
    hdr[5] = 1.0  # iform for 2D image
    hdr[12] = float(nsam)  # number of pixels per line
    hdr[13] = float(labrec)  # number of records in file header
    hdr[22] = float(labbyt)  # total number of bytes in header
    hdr[23] = float(lenbyt)  # record length in bytes

    # adjust for Fortran indexing
    hdr = hdr[1:]
    hdr.append(0.0)
    # pack binary data into a string
    return [struct.pack("f", v) for v in hdr]


def _save(im, fp, filename):
    if im.mode[0] != "F":
        im = im.convert("F")

    hdr = makeSpiderHeader(im)
    if len(hdr) < 256:
        msg = "Error creating Spider header"
        raise OSError(msg)

    # write the SPIDER header
    fp.writelines(hdr)

    rawmode = "F;32NF"  # 32-bit native floating point
    ImageFile._save(im, fp, [("raw", (0, 0) + im.size, 0, (rawmode, 0, 1))])


def _save_spider(im, fp, filename):
    # get the filename extension and register it with Image
    ext = os.path.splitext(filename)[1]
    Image.register_extension(SpiderImageFile.format, ext)
    _save(im, fp, filename)


# --------------------------------------------------------------------


Image.register_open(SpiderImageFile.format, SpiderImageFile)
Image.register_save(SpiderImageFile.format, _save_spider)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Syntax: python3 SpiderImagePlugin.py [infile] [outfile]")
        sys.exit()

    filename = sys.argv[1]
    if not isSpiderImage(filename):
        print("input image must be in Spider format")
        sys.exit()

    with Image.open(filename) as im:
        print("image: " + str(im))
        print("format: " + str(im.format))
        print("size: " + str(im.size))
        print("mode: " + str(im.mode))
        print("max, min: ", end=" ")
        print(im.getextrema())

        if len(sys.argv) > 2:
            outfile = sys.argv[2]

            # perform some image operation
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            print(
                f"saving a flipped version of {os.path.basename(filename)} "
                f"as {outfile} "
            )
            im.save(outfile, SpiderImageFile.format)
