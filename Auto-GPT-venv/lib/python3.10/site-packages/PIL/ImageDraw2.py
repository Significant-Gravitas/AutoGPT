#
# The Python Imaging Library
# $Id$
#
# WCK-style drawing interface operations
#
# History:
# 2003-12-07 fl   created
# 2005-05-15 fl   updated; added to PIL as ImageDraw2
# 2005-05-15 fl   added text support
# 2005-05-20 fl   added arc/chord/pieslice support
#
# Copyright (c) 2003-2005 by Secret Labs AB
# Copyright (c) 2003-2005 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#


"""
(Experimental) WCK-style drawing interface operations

.. seealso:: :py:mod:`PIL.ImageDraw`
"""


import warnings

from . import Image, ImageColor, ImageDraw, ImageFont, ImagePath
from ._deprecate import deprecate


class Pen:
    """Stores an outline color and width."""

    def __init__(self, color, width=1, opacity=255):
        self.color = ImageColor.getrgb(color)
        self.width = width


class Brush:
    """Stores a fill color"""

    def __init__(self, color, opacity=255):
        self.color = ImageColor.getrgb(color)


class Font:
    """Stores a TrueType font and color"""

    def __init__(self, color, file, size=12):
        # FIXME: add support for bitmap fonts
        self.color = ImageColor.getrgb(color)
        self.font = ImageFont.truetype(file, size)


class Draw:
    """
    (Experimental) WCK-style drawing interface
    """

    def __init__(self, image, size=None, color=None):
        if not hasattr(image, "im"):
            image = Image.new(image, size, color)
        self.draw = ImageDraw.Draw(image)
        self.image = image
        self.transform = None

    def flush(self):
        return self.image

    def render(self, op, xy, pen, brush=None):
        # handle color arguments
        outline = fill = None
        width = 1
        if isinstance(pen, Pen):
            outline = pen.color
            width = pen.width
        elif isinstance(brush, Pen):
            outline = brush.color
            width = brush.width
        if isinstance(brush, Brush):
            fill = brush.color
        elif isinstance(pen, Brush):
            fill = pen.color
        # handle transformation
        if self.transform:
            xy = ImagePath.Path(xy)
            xy.transform(self.transform)
        # render the item
        if op == "line":
            self.draw.line(xy, fill=outline, width=width)
        else:
            getattr(self.draw, op)(xy, fill=fill, outline=outline)

    def settransform(self, offset):
        """Sets a transformation offset."""
        (xoffset, yoffset) = offset
        self.transform = (1, 0, xoffset, 0, 1, yoffset)

    def arc(self, xy, start, end, *options):
        """
        Draws an arc (a portion of a circle outline) between the start and end
        angles, inside the given bounding box.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.arc`
        """
        self.render("arc", xy, start, end, *options)

    def chord(self, xy, start, end, *options):
        """
        Same as :py:meth:`~PIL.ImageDraw2.Draw.arc`, but connects the end points
        with a straight line.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.chord`
        """
        self.render("chord", xy, start, end, *options)

    def ellipse(self, xy, *options):
        """
        Draws an ellipse inside the given bounding box.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.ellipse`
        """
        self.render("ellipse", xy, *options)

    def line(self, xy, *options):
        """
        Draws a line between the coordinates in the ``xy`` list.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.line`
        """
        self.render("line", xy, *options)

    def pieslice(self, xy, start, end, *options):
        """
        Same as arc, but also draws straight lines between the end points and the
        center of the bounding box.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.pieslice`
        """
        self.render("pieslice", xy, start, end, *options)

    def polygon(self, xy, *options):
        """
        Draws a polygon.

        The polygon outline consists of straight lines between the given
        coordinates, plus a straight line between the last and the first
        coordinate.


        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.polygon`
        """
        self.render("polygon", xy, *options)

    def rectangle(self, xy, *options):
        """
        Draws a rectangle.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.rectangle`
        """
        self.render("rectangle", xy, *options)

    def text(self, xy, text, font):
        """
        Draws the string at the given position.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.text`
        """
        if self.transform:
            xy = ImagePath.Path(xy)
            xy.transform(self.transform)
        self.draw.text(xy, text, font=font.font, fill=font.color)

    def textsize(self, text, font):
        """
        .. deprecated:: 9.2.0

        Return the size of the given string, in pixels.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textsize`
        """
        deprecate("textsize", 10, "textbbox or textlength")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return self.draw.textsize(text, font=font.font)

    def textbbox(self, xy, text, font):
        """
        Returns bounding box (in pixels) of given text.

        :return: ``(left, top, right, bottom)`` bounding box

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textbbox`
        """
        if self.transform:
            xy = ImagePath.Path(xy)
            xy.transform(self.transform)
        return self.draw.textbbox(xy, text, font=font.font)

    def textlength(self, text, font):
        """
        Returns length (in pixels) of given text.
        This is the amount by which following text should be offset.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textlength`
        """
        return self.draw.textlength(text, font=font.font)
