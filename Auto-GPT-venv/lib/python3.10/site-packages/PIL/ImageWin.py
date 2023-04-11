#
# The Python Imaging Library.
# $Id$
#
# a Windows DIB display interface
#
# History:
# 1996-05-20 fl   Created
# 1996-09-20 fl   Fixed subregion exposure
# 1997-09-21 fl   Added draw primitive (for tzPrint)
# 2003-05-21 fl   Added experimental Window/ImageWindow classes
# 2003-09-05 fl   Added fromstring/tostring methods
#
# Copyright (c) Secret Labs AB 1997-2003.
# Copyright (c) Fredrik Lundh 1996-2003.
#
# See the README file for information on usage and redistribution.
#

from . import Image


class HDC:
    """
    Wraps an HDC integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods.
    """

    def __init__(self, dc):
        self.dc = dc

    def __int__(self):
        return self.dc


class HWND:
    """
    Wraps an HWND integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods, instead of a DC.
    """

    def __init__(self, wnd):
        self.wnd = wnd

    def __int__(self):
        return self.wnd


class Dib:
    """
    A Windows bitmap with the given mode and size.  The mode can be one of "1",
    "L", "P", or "RGB".

    If the display requires a palette, this constructor creates a suitable
    palette and associates it with the image. For an "L" image, 128 greylevels
    are allocated. For an "RGB" image, a 6x6x6 colour cube is used, together
    with 20 greylevels.

    To make sure that palettes work properly under Windows, you must call the
    ``palette`` method upon certain events from Windows.

    :param image: Either a PIL image, or a mode string. If a mode string is
                  used, a size must also be given.  The mode can be one of "1",
                  "L", "P", or "RGB".
    :param size: If the first argument is a mode string, this
                 defines the size of the image.
    """

    def __init__(self, image, size=None):
        if hasattr(image, "mode") and hasattr(image, "size"):
            mode = image.mode
            size = image.size
        else:
            mode = image
            image = None
        if mode not in ["1", "L", "P", "RGB"]:
            mode = Image.getmodebase(mode)
        self.image = Image.core.display(mode, size)
        self.mode = mode
        self.size = size
        if image:
            self.paste(image)

    def expose(self, handle):
        """
        Copy the bitmap contents to a device context.

        :param handle: Device context (HDC), cast to a Python integer, or an
                       HDC or HWND instance.  In PythonWin, you can use
                       ``CDC.GetHandleAttrib()`` to get a suitable handle.
        """
        if isinstance(handle, HWND):
            dc = self.image.getdc(handle)
            try:
                result = self.image.expose(dc)
            finally:
                self.image.releasedc(handle, dc)
        else:
            result = self.image.expose(handle)
        return result

    def draw(self, handle, dst, src=None):
        """
        Same as expose, but allows you to specify where to draw the image, and
        what part of it to draw.

        The destination and source areas are given as 4-tuple rectangles. If
        the source is omitted, the entire image is copied. If the source and
        the destination have different sizes, the image is resized as
        necessary.
        """
        if not src:
            src = (0, 0) + self.size
        if isinstance(handle, HWND):
            dc = self.image.getdc(handle)
            try:
                result = self.image.draw(dc, dst, src)
            finally:
                self.image.releasedc(handle, dc)
        else:
            result = self.image.draw(handle, dst, src)
        return result

    def query_palette(self, handle):
        """
        Installs the palette associated with the image in the given device
        context.

        This method should be called upon **QUERYNEWPALETTE** and
        **PALETTECHANGED** events from Windows. If this method returns a
        non-zero value, one or more display palette entries were changed, and
        the image should be redrawn.

        :param handle: Device context (HDC), cast to a Python integer, or an
                       HDC or HWND instance.
        :return: A true value if one or more entries were changed (this
                 indicates that the image should be redrawn).
        """
        if isinstance(handle, HWND):
            handle = self.image.getdc(handle)
            try:
                result = self.image.query_palette(handle)
            finally:
                self.image.releasedc(handle, handle)
        else:
            result = self.image.query_palette(handle)
        return result

    def paste(self, im, box=None):
        """
        Paste a PIL image into the bitmap image.

        :param im: A PIL image.  The size must match the target region.
                   If the mode does not match, the image is converted to the
                   mode of the bitmap image.
        :param box: A 4-tuple defining the left, upper, right, and
                    lower pixel coordinate.  See :ref:`coordinate-system`. If
                    None is given instead of a tuple, all of the image is
                    assumed.
        """
        im.load()
        if self.mode != im.mode:
            im = im.convert(self.mode)
        if box:
            self.image.paste(im.im, box)
        else:
            self.image.paste(im.im)

    def frombytes(self, buffer):
        """
        Load display memory contents from byte data.

        :param buffer: A buffer containing display data (usually
                       data returned from :py:func:`~PIL.ImageWin.Dib.tobytes`)
        """
        return self.image.frombytes(buffer)

    def tobytes(self):
        """
        Copy display memory contents to bytes object.

        :return: A bytes object containing display data.
        """
        return self.image.tobytes()


class Window:
    """Create a Window with the given title size."""

    def __init__(self, title="PIL", width=None, height=None):
        self.hwnd = Image.core.createwindow(
            title, self.__dispatcher, width or 0, height or 0
        )

    def __dispatcher(self, action, *args):
        return getattr(self, "ui_handle_" + action)(*args)

    def ui_handle_clear(self, dc, x0, y0, x1, y1):
        pass

    def ui_handle_damage(self, x0, y0, x1, y1):
        pass

    def ui_handle_destroy(self):
        pass

    def ui_handle_repair(self, dc, x0, y0, x1, y1):
        pass

    def ui_handle_resize(self, width, height):
        pass

    def mainloop(self):
        Image.core.eventloop()


class ImageWindow(Window):
    """Create an image window which displays the given image."""

    def __init__(self, image, title="PIL"):
        if not isinstance(image, Dib):
            image = Dib(image)
        self.image = image
        width, height = image.size
        super().__init__(title, width=width, height=height)

    def ui_handle_repair(self, dc, x0, y0, x1, y1):
        self.image.draw(dc, (x0, y0, x1, y1))
