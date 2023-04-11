#
# The Python Imaging Library.
# $Id$
#
# a Tk display interface
#
# History:
# 96-04-08 fl   Created
# 96-09-06 fl   Added getimage method
# 96-11-01 fl   Rewritten, removed image attribute and crop method
# 97-05-09 fl   Use PyImagingPaste method instead of image type
# 97-05-12 fl   Minor tweaks to match the IFUNC95 interface
# 97-05-17 fl   Support the "pilbitmap" booster patch
# 97-06-05 fl   Added file= and data= argument to image constructors
# 98-03-09 fl   Added width and height methods to Image classes
# 98-07-02 fl   Use default mode for "P" images without palette attribute
# 98-07-02 fl   Explicitly destroy Tkinter image objects
# 99-07-24 fl   Support multiple Tk interpreters (from Greg Couch)
# 99-07-26 fl   Automatically hook into Tkinter (if possible)
# 99-08-15 fl   Hook uses _imagingtk instead of _imaging
#
# Copyright (c) 1997-1999 by Secret Labs AB
# Copyright (c) 1996-1997 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import tkinter
from io import BytesIO

from . import Image
from ._deprecate import deprecate

# --------------------------------------------------------------------
# Check for Tkinter interface hooks

_pilbitmap_ok = None


def _pilbitmap_check():
    global _pilbitmap_ok
    if _pilbitmap_ok is None:
        try:
            im = Image.new("1", (1, 1))
            tkinter.BitmapImage(data=f"PIL:{im.im.id}")
            _pilbitmap_ok = 1
        except tkinter.TclError:
            _pilbitmap_ok = 0
    return _pilbitmap_ok


def _get_image_from_kw(kw):
    source = None
    if "file" in kw:
        source = kw.pop("file")
    elif "data" in kw:
        source = BytesIO(kw.pop("data"))
    if source:
        return Image.open(source)


def _pyimagingtkcall(command, photo, id):
    tk = photo.tk
    try:
        tk.call(command, photo, id)
    except tkinter.TclError:
        # activate Tkinter hook
        # may raise an error if it cannot attach to Tkinter
        from . import _imagingtk

        _imagingtk.tkinit(tk.interpaddr())
        tk.call(command, photo, id)


# --------------------------------------------------------------------
# PhotoImage


class PhotoImage:
    """
    A Tkinter-compatible photo image.  This can be used
    everywhere Tkinter expects an image object.  If the image is an RGBA
    image, pixels having alpha 0 are treated as transparent.

    The constructor takes either a PIL image, or a mode and a size.
    Alternatively, you can use the ``file`` or ``data`` options to initialize
    the photo image object.

    :param image: Either a PIL image, or a mode string.  If a mode string is
                  used, a size must also be given.
    :param size: If the first argument is a mode string, this defines the size
                 of the image.
    :keyword file: A filename to load the image from (using
                   ``Image.open(file)``).
    :keyword data: An 8-bit string containing image data (as loaded from an
                   image file).
    """

    def __init__(self, image=None, size=None, **kw):
        # Tk compatibility: file or data
        if image is None:
            image = _get_image_from_kw(kw)

        if hasattr(image, "mode") and hasattr(image, "size"):
            # got an image instead of a mode
            mode = image.mode
            if mode == "P":
                # palette mapped data
                image.apply_transparency()
                image.load()
                try:
                    mode = image.palette.mode
                except AttributeError:
                    mode = "RGB"  # default
            size = image.size
            kw["width"], kw["height"] = size
        else:
            mode = image
            image = None

        if mode not in ["1", "L", "RGB", "RGBA"]:
            mode = Image.getmodebase(mode)

        self.__mode = mode
        self.__size = size
        self.__photo = tkinter.PhotoImage(**kw)
        self.tk = self.__photo.tk
        if image:
            self.paste(image)

    def __del__(self):
        name = self.__photo.name
        self.__photo.name = None
        try:
            self.__photo.tk.call("image", "delete", name)
        except Exception:
            pass  # ignore internal errors

    def __str__(self):
        """
        Get the Tkinter photo image identifier.  This method is automatically
        called by Tkinter whenever a PhotoImage object is passed to a Tkinter
        method.

        :return: A Tkinter photo image identifier (a string).
        """
        return str(self.__photo)

    def width(self):
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        return self.__size[0]

    def height(self):
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        return self.__size[1]

    def paste(self, im, box=None):
        """
        Paste a PIL image into the photo image.  Note that this can
        be very slow if the photo image is displayed.

        :param im: A PIL image. The size must match the target region.  If the
                   mode does not match, the image is converted to the mode of
                   the bitmap image.
        :param box: Deprecated. This parameter will be removed in Pillow 10
                    (2023-07-01).
        """

        if box is not None:
            deprecate("The box parameter", 10, None)

        # convert to blittable
        im.load()
        image = im.im
        if image.isblock() and im.mode == self.__mode:
            block = image
        else:
            block = image.new_block(self.__mode, im.size)
            image.convert2(block, image)  # convert directly between buffers

        _pyimagingtkcall("PyImagingPhoto", self.__photo, block.id)


# --------------------------------------------------------------------
# BitmapImage


class BitmapImage:
    """
    A Tkinter-compatible bitmap image.  This can be used everywhere Tkinter
    expects an image object.

    The given image must have mode "1".  Pixels having value 0 are treated as
    transparent.  Options, if any, are passed on to Tkinter.  The most commonly
    used option is ``foreground``, which is used to specify the color for the
    non-transparent parts.  See the Tkinter documentation for information on
    how to specify colours.

    :param image: A PIL image.
    """

    def __init__(self, image=None, **kw):
        # Tk compatibility: file or data
        if image is None:
            image = _get_image_from_kw(kw)

        self.__mode = image.mode
        self.__size = image.size

        if _pilbitmap_check():
            # fast way (requires the pilbitmap booster patch)
            image.load()
            kw["data"] = f"PIL:{image.im.id}"
            self.__im = image  # must keep a reference
        else:
            # slow but safe way
            kw["data"] = image.tobitmap()
        self.__photo = tkinter.BitmapImage(**kw)

    def __del__(self):
        name = self.__photo.name
        self.__photo.name = None
        try:
            self.__photo.tk.call("image", "delete", name)
        except Exception:
            pass  # ignore internal errors

    def width(self):
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        return self.__size[0]

    def height(self):
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        return self.__size[1]

    def __str__(self):
        """
        Get the Tkinter bitmap image identifier.  This method is automatically
        called by Tkinter whenever a BitmapImage object is passed to a Tkinter
        method.

        :return: A Tkinter bitmap image identifier (a string).
        """
        return str(self.__photo)


def getimage(photo):
    """Copies the contents of a PhotoImage to a PIL image memory."""
    im = Image.new("RGBA", (photo.width(), photo.height()))
    block = im.im

    _pyimagingtkcall("PyImagingPhotoGet", photo, block.id)

    return im


def _show(image, title):
    """Helper for the Image.show method."""

    class UI(tkinter.Label):
        def __init__(self, master, im):
            if im.mode == "1":
                self.image = BitmapImage(im, foreground="white", master=master)
            else:
                self.image = PhotoImage(im, master=master)
            super().__init__(master, image=self.image, bg="black", bd=0)

    if not tkinter._default_root:
        msg = "tkinter not initialized"
        raise OSError(msg)
    top = tkinter.Toplevel()
    if title:
        top.title(title)
    UI(top, image).pack()
