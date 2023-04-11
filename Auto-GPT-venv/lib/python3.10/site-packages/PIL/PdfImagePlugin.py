#
# The Python Imaging Library.
# $Id$
#
# PDF (Acrobat) file handling
#
# History:
# 1996-07-16 fl   Created
# 1997-01-18 fl   Fixed header
# 2004-02-21 fl   Fixes for 1/L/CMYK images, etc.
# 2004-02-24 fl   Fixes for 1 and P images.
#
# Copyright (c) 1997-2004 by Secret Labs AB.  All rights reserved.
# Copyright (c) 1996-1997 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

##
# Image plugin for PDF images (output only).
##

import io
import math
import os
import time

from . import Image, ImageFile, ImageSequence, PdfParser, __version__, features

#
# --------------------------------------------------------------------

# object ids:
#  1. catalogue
#  2. pages
#  3. image
#  4. page
#  5. page contents


def _save_all(im, fp, filename):
    _save(im, fp, filename, save_all=True)


##
# (Internal) Image save plugin for the PDF format.


def _save(im, fp, filename, save_all=False):
    is_appending = im.encoderinfo.get("append", False)
    if is_appending:
        existing_pdf = PdfParser.PdfParser(f=fp, filename=filename, mode="r+b")
    else:
        existing_pdf = PdfParser.PdfParser(f=fp, filename=filename, mode="w+b")

    dpi = im.encoderinfo.get("dpi")
    if dpi:
        x_resolution = dpi[0]
        y_resolution = dpi[1]
    else:
        x_resolution = y_resolution = im.encoderinfo.get("resolution", 72.0)

    info = {
        "title": None
        if is_appending
        else os.path.splitext(os.path.basename(filename))[0],
        "author": None,
        "subject": None,
        "keywords": None,
        "creator": None,
        "producer": None,
        "creationDate": None if is_appending else time.gmtime(),
        "modDate": None if is_appending else time.gmtime(),
    }
    for k, default in info.items():
        v = im.encoderinfo.get(k) if k in im.encoderinfo else default
        if v:
            existing_pdf.info[k[0].upper() + k[1:]] = v

    #
    # make sure image data is available
    im.load()

    existing_pdf.start_writing()
    existing_pdf.write_header()
    existing_pdf.write_comment(f"created by Pillow {__version__} PDF driver")

    #
    # pages
    ims = [im]
    if save_all:
        append_images = im.encoderinfo.get("append_images", [])
        for append_im in append_images:
            append_im.encoderinfo = im.encoderinfo.copy()
            ims.append(append_im)
    number_of_pages = 0
    image_refs = []
    page_refs = []
    contents_refs = []
    for im in ims:
        im_number_of_pages = 1
        if save_all:
            try:
                im_number_of_pages = im.n_frames
            except AttributeError:
                # Image format does not have n_frames.
                # It is a single frame image
                pass
        number_of_pages += im_number_of_pages
        for i in range(im_number_of_pages):
            image_refs.append(existing_pdf.next_object_id(0))
            page_refs.append(existing_pdf.next_object_id(0))
            contents_refs.append(existing_pdf.next_object_id(0))
            existing_pdf.pages.append(page_refs[-1])

    #
    # catalog and list of pages
    existing_pdf.write_catalog()

    page_number = 0
    for im_sequence in ims:
        im_pages = ImageSequence.Iterator(im_sequence) if save_all else [im_sequence]
        for im in im_pages:
            # FIXME: Should replace ASCIIHexDecode with RunLengthDecode
            # (packbits) or LZWDecode (tiff/lzw compression).  Note that
            # PDF 1.2 also supports Flatedecode (zip compression).

            bits = 8
            params = None
            decode = None

            #
            # Get image characteristics

            width, height = im.size

            if im.mode == "1":
                if features.check("libtiff"):
                    filter = "CCITTFaxDecode"
                    bits = 1
                    params = PdfParser.PdfArray(
                        [
                            PdfParser.PdfDict(
                                {
                                    "K": -1,
                                    "BlackIs1": True,
                                    "Columns": width,
                                    "Rows": height,
                                }
                            )
                        ]
                    )
                else:
                    filter = "DCTDecode"
                colorspace = PdfParser.PdfName("DeviceGray")
                procset = "ImageB"  # grayscale
            elif im.mode == "L":
                filter = "DCTDecode"
                # params = f"<< /Predictor 15 /Columns {width-2} >>"
                colorspace = PdfParser.PdfName("DeviceGray")
                procset = "ImageB"  # grayscale
            elif im.mode == "P":
                filter = "ASCIIHexDecode"
                palette = im.getpalette()
                colorspace = [
                    PdfParser.PdfName("Indexed"),
                    PdfParser.PdfName("DeviceRGB"),
                    255,
                    PdfParser.PdfBinary(palette),
                ]
                procset = "ImageI"  # indexed color
            elif im.mode == "RGB":
                filter = "DCTDecode"
                colorspace = PdfParser.PdfName("DeviceRGB")
                procset = "ImageC"  # color images
            elif im.mode == "RGBA":
                filter = "JPXDecode"
                colorspace = PdfParser.PdfName("DeviceRGB")
                procset = "ImageC"  # color images
            elif im.mode == "CMYK":
                filter = "DCTDecode"
                colorspace = PdfParser.PdfName("DeviceCMYK")
                procset = "ImageC"  # color images
                decode = [1, 0, 1, 0, 1, 0, 1, 0]
            else:
                msg = f"cannot save mode {im.mode}"
                raise ValueError(msg)

            #
            # image

            op = io.BytesIO()

            if filter == "ASCIIHexDecode":
                ImageFile._save(im, op, [("hex", (0, 0) + im.size, 0, im.mode)])
            elif filter == "CCITTFaxDecode":
                im.save(
                    op,
                    "TIFF",
                    compression="group4",
                    # use a single strip
                    strip_size=math.ceil(im.width / 8) * im.height,
                )
            elif filter == "DCTDecode":
                Image.SAVE["JPEG"](im, op, filename)
            elif filter == "JPXDecode":
                Image.SAVE["JPEG2000"](im, op, filename)
            elif filter == "FlateDecode":
                ImageFile._save(im, op, [("zip", (0, 0) + im.size, 0, im.mode)])
            elif filter == "RunLengthDecode":
                ImageFile._save(im, op, [("packbits", (0, 0) + im.size, 0, im.mode)])
            else:
                msg = f"unsupported PDF filter ({filter})"
                raise ValueError(msg)

            stream = op.getvalue()
            if filter == "CCITTFaxDecode":
                stream = stream[8:]
                filter = PdfParser.PdfArray([PdfParser.PdfName(filter)])
            else:
                filter = PdfParser.PdfName(filter)

            existing_pdf.write_obj(
                image_refs[page_number],
                stream=stream,
                Type=PdfParser.PdfName("XObject"),
                Subtype=PdfParser.PdfName("Image"),
                Width=width,  # * 72.0 / x_resolution,
                Height=height,  # * 72.0 / y_resolution,
                Filter=filter,
                BitsPerComponent=bits,
                Decode=decode,
                DecodeParms=params,
                ColorSpace=colorspace,
            )

            #
            # page

            existing_pdf.write_page(
                page_refs[page_number],
                Resources=PdfParser.PdfDict(
                    ProcSet=[PdfParser.PdfName("PDF"), PdfParser.PdfName(procset)],
                    XObject=PdfParser.PdfDict(image=image_refs[page_number]),
                ),
                MediaBox=[
                    0,
                    0,
                    width * 72.0 / x_resolution,
                    height * 72.0 / y_resolution,
                ],
                Contents=contents_refs[page_number],
            )

            #
            # page contents

            page_contents = b"q %f 0 0 %f 0 0 cm /image Do Q\n" % (
                width * 72.0 / x_resolution,
                height * 72.0 / y_resolution,
            )

            existing_pdf.write_obj(contents_refs[page_number], stream=page_contents)

            page_number += 1

    #
    # trailer
    existing_pdf.write_xref_and_trailer()
    if hasattr(fp, "flush"):
        fp.flush()
    existing_pdf.close()


#
# --------------------------------------------------------------------


Image.register_save("PDF", _save)
Image.register_save_all("PDF", _save_all)

Image.register_extension("PDF", ".pdf")

Image.register_mime("PDF", "application/pdf")
