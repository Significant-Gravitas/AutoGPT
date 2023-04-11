#
# The Python Imaging Library.
# $Id$
#
# standard filters
#
# History:
# 1995-11-27 fl   Created
# 2002-06-08 fl   Added rank and mode filters
# 2003-09-15 fl   Fixed rank calculation in rank filter; added expand call
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2002 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
import functools


class Filter:
    pass


class MultibandFilter(Filter):
    pass


class BuiltinFilter(MultibandFilter):
    def filter(self, image):
        if image.mode == "P":
            msg = "cannot filter palette images"
            raise ValueError(msg)
        return image.filter(*self.filterargs)


class Kernel(BuiltinFilter):
    """
    Create a convolution kernel.  The current version only
    supports 3x3 and 5x5 integer and floating point kernels.

    In the current version, kernels can only be applied to
    "L" and "RGB" images.

    :param size: Kernel size, given as (width, height). In the current
                    version, this must be (3,3) or (5,5).
    :param kernel: A sequence containing kernel weights.
    :param scale: Scale factor. If given, the result for each pixel is
                    divided by this value.  The default is the sum of the
                    kernel weights.
    :param offset: Offset. If given, this value is added to the result,
                    after it has been divided by the scale factor.
    """

    name = "Kernel"

    def __init__(self, size, kernel, scale=None, offset=0):
        if scale is None:
            # default scale is sum of kernel
            scale = functools.reduce(lambda a, b: a + b, kernel)
        if size[0] * size[1] != len(kernel):
            msg = "not enough coefficients in kernel"
            raise ValueError(msg)
        self.filterargs = size, scale, offset, kernel


class RankFilter(Filter):
    """
    Create a rank filter.  The rank filter sorts all pixels in
    a window of the given size, and returns the ``rank``'th value.

    :param size: The kernel size, in pixels.
    :param rank: What pixel value to pick.  Use 0 for a min filter,
                 ``size * size / 2`` for a median filter, ``size * size - 1``
                 for a max filter, etc.
    """

    name = "Rank"

    def __init__(self, size, rank):
        self.size = size
        self.rank = rank

    def filter(self, image):
        if image.mode == "P":
            msg = "cannot filter palette images"
            raise ValueError(msg)
        image = image.expand(self.size // 2, self.size // 2)
        return image.rankfilter(self.size, self.rank)


class MedianFilter(RankFilter):
    """
    Create a median filter. Picks the median pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """

    name = "Median"

    def __init__(self, size=3):
        self.size = size
        self.rank = size * size // 2


class MinFilter(RankFilter):
    """
    Create a min filter.  Picks the lowest pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """

    name = "Min"

    def __init__(self, size=3):
        self.size = size
        self.rank = 0


class MaxFilter(RankFilter):
    """
    Create a max filter.  Picks the largest pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """

    name = "Max"

    def __init__(self, size=3):
        self.size = size
        self.rank = size * size - 1


class ModeFilter(Filter):
    """
    Create a mode filter. Picks the most frequent pixel value in a box with the
    given size.  Pixel values that occur only once or twice are ignored; if no
    pixel value occurs more than twice, the original pixel value is preserved.

    :param size: The kernel size, in pixels.
    """

    name = "Mode"

    def __init__(self, size=3):
        self.size = size

    def filter(self, image):
        return image.modefilter(self.size)


class GaussianBlur(MultibandFilter):
    """Blurs the image with a sequence of extended box filters, which
    approximates a Gaussian kernel. For details on accuracy see
    <https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf>

    :param radius: Standard deviation of the Gaussian kernel.
    """

    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)


class BoxBlur(MultibandFilter):
    """Blurs the image by setting each pixel to the average value of the pixels
    in a square box extending radius pixels in each direction.
    Supports float radius of arbitrary size. Uses an optimized implementation
    which runs in linear time relative to the size of the image
    for any radius value.

    :param radius: Size of the box in one direction. Radius 0 does not blur,
                   returns an identical image. Radius 1 takes 1 pixel
                   in each direction, i.e. 9 pixels in total.
    """

    name = "BoxBlur"

    def __init__(self, radius):
        if radius < 0:
            msg = "radius must be >= 0"
            raise ValueError(msg)
        self.radius = radius

    def filter(self, image):
        return image.box_blur(self.radius)


class UnsharpMask(MultibandFilter):
    """Unsharp mask filter.

    See Wikipedia's entry on `digital unsharp masking`_ for an explanation of
    the parameters.

    :param radius: Blur Radius
    :param percent: Unsharp strength, in percent
    :param threshold: Threshold controls the minimum brightness change that
      will be sharpened

    .. _digital unsharp masking: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking

    """  # noqa: E501

    name = "UnsharpMask"

    def __init__(self, radius=2, percent=150, threshold=3):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def filter(self, image):
        return image.unsharp_mask(self.radius, self.percent, self.threshold)


class BLUR(BuiltinFilter):
    name = "Blur"
    # fmt: off
    filterargs = (5, 5), 16, 0, (
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
    )
    # fmt: on


class CONTOUR(BuiltinFilter):
    name = "Contour"
    # fmt: off
    filterargs = (3, 3), 1, 255, (
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1,
    )
    # fmt: on


class DETAIL(BuiltinFilter):
    name = "Detail"
    # fmt: off
    filterargs = (3, 3), 6, 0, (
        0,  -1,  0,
        -1, 10, -1,
        0,  -1,  0,
    )
    # fmt: on


class EDGE_ENHANCE(BuiltinFilter):
    name = "Edge-enhance"
    # fmt: off
    filterargs = (3, 3), 2, 0, (
        -1, -1, -1,
        -1, 10, -1,
        -1, -1, -1,
    )
    # fmt: on


class EDGE_ENHANCE_MORE(BuiltinFilter):
    name = "Edge-enhance More"
    # fmt: off
    filterargs = (3, 3), 1, 0, (
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1,
    )
    # fmt: on


class EMBOSS(BuiltinFilter):
    name = "Emboss"
    # fmt: off
    filterargs = (3, 3), 1, 128, (
        -1, 0, 0,
        0,  1, 0,
        0,  0, 0,
    )
    # fmt: on


class FIND_EDGES(BuiltinFilter):
    name = "Find Edges"
    # fmt: off
    filterargs = (3, 3), 1, 0, (
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1,
    )
    # fmt: on


class SHARPEN(BuiltinFilter):
    name = "Sharpen"
    # fmt: off
    filterargs = (3, 3), 16, 0, (
        -2, -2, -2,
        -2, 32, -2,
        -2, -2, -2,
    )
    # fmt: on


class SMOOTH(BuiltinFilter):
    name = "Smooth"
    # fmt: off
    filterargs = (3, 3), 13, 0, (
        1, 1, 1,
        1, 5, 1,
        1, 1, 1,
    )
    # fmt: on


class SMOOTH_MORE(BuiltinFilter):
    name = "Smooth More"
    # fmt: off
    filterargs = (5, 5), 100, 0, (
        1, 1,  1, 1, 1,
        1, 5,  5, 5, 1,
        1, 5, 44, 5, 1,
        1, 5,  5, 5, 1,
        1, 1,  1, 1, 1,
    )
    # fmt: on


class Color3DLUT(MultibandFilter):
    """Three-dimensional color lookup table.

    Transforms 3-channel pixels using the values of the channels as coordinates
    in the 3D lookup table and interpolating the nearest elements.

    This method allows you to apply almost any color transformation
    in constant time by using pre-calculated decimated tables.

    .. versionadded:: 5.2.0

    :param size: Size of the table. One int or tuple of (int, int, int).
                 Minimal size in any dimension is 2, maximum is 65.
    :param table: Flat lookup table. A list of ``channels * size**3``
                  float elements or a list of ``size**3`` channels-sized
                  tuples with floats. Channels are changed first,
                  then first dimension, then second, then third.
                  Value 0.0 corresponds lowest value of output, 1.0 highest.
    :param channels: Number of channels in the table. Could be 3 or 4.
                     Default is 3.
    :param target_mode: A mode for the result image. Should have not less
                        than ``channels`` channels. Default is ``None``,
                        which means that mode wouldn't be changed.
    """

    name = "Color 3D LUT"

    def __init__(self, size, table, channels=3, target_mode=None, **kwargs):
        if channels not in (3, 4):
            msg = "Only 3 or 4 output channels are supported"
            raise ValueError(msg)
        self.size = size = self._check_size(size)
        self.channels = channels
        self.mode = target_mode

        # Hidden flag `_copy_table=False` could be used to avoid extra copying
        # of the table if the table is specially made for the constructor.
        copy_table = kwargs.get("_copy_table", True)
        items = size[0] * size[1] * size[2]
        wrong_size = False

        numpy = None
        if hasattr(table, "shape"):
            try:
                import numpy
            except ImportError:  # pragma: no cover
                pass

        if numpy and isinstance(table, numpy.ndarray):
            if copy_table:
                table = table.copy()

            if table.shape in [
                (items * channels,),
                (items, channels),
                (size[2], size[1], size[0], channels),
            ]:
                table = table.reshape(items * channels)
            else:
                wrong_size = True

        else:
            if copy_table:
                table = list(table)

            # Convert to a flat list
            if table and isinstance(table[0], (list, tuple)):
                table, raw_table = [], table
                for pixel in raw_table:
                    if len(pixel) != channels:
                        msg = (
                            "The elements of the table should "
                            f"have a length of {channels}."
                        )
                        raise ValueError(msg)
                    table.extend(pixel)

        if wrong_size or len(table) != items * channels:
            msg = (
                "The table should have either channels * size**3 float items "
                "or size**3 items of channels-sized tuples with floats. "
                f"Table should be: {channels}x{size[0]}x{size[1]}x{size[2]}. "
                f"Actual length: {len(table)}"
            )
            raise ValueError(msg)
        self.table = table

    @staticmethod
    def _check_size(size):
        try:
            _, _, _ = size
        except ValueError as e:
            msg = "Size should be either an integer or a tuple of three integers."
            raise ValueError(msg) from e
        except TypeError:
            size = (size, size, size)
        size = [int(x) for x in size]
        for size_1d in size:
            if not 2 <= size_1d <= 65:
                msg = "Size should be in [2, 65] range."
                raise ValueError(msg)
        return size

    @classmethod
    def generate(cls, size, callback, channels=3, target_mode=None):
        """Generates new LUT using provided callback.

        :param size: Size of the table. Passed to the constructor.
        :param callback: Function with three parameters which correspond
                         three color channels. Will be called ``size**3``
                         times with values from 0.0 to 1.0 and should return
                         a tuple with ``channels`` elements.
        :param channels: The number of channels which should return callback.
        :param target_mode: Passed to the constructor of the resulting
                            lookup table.
        """
        size_1d, size_2d, size_3d = cls._check_size(size)
        if channels not in (3, 4):
            msg = "Only 3 or 4 output channels are supported"
            raise ValueError(msg)

        table = [0] * (size_1d * size_2d * size_3d * channels)
        idx_out = 0
        for b in range(size_3d):
            for g in range(size_2d):
                for r in range(size_1d):
                    table[idx_out : idx_out + channels] = callback(
                        r / (size_1d - 1), g / (size_2d - 1), b / (size_3d - 1)
                    )
                    idx_out += channels

        return cls(
            (size_1d, size_2d, size_3d),
            table,
            channels=channels,
            target_mode=target_mode,
            _copy_table=False,
        )

    def transform(self, callback, with_normals=False, channels=None, target_mode=None):
        """Transforms the table values using provided callback and returns
        a new LUT with altered values.

        :param callback: A function which takes old lookup table values
                         and returns a new set of values. The number
                         of arguments which function should take is
                         ``self.channels`` or ``3 + self.channels``
                         if ``with_normals`` flag is set.
                         Should return a tuple of ``self.channels`` or
                         ``channels`` elements if it is set.
        :param with_normals: If true, ``callback`` will be called with
                             coordinates in the color cube as the first
                             three arguments. Otherwise, ``callback``
                             will be called only with actual color values.
        :param channels: The number of channels in the resulting lookup table.
        :param target_mode: Passed to the constructor of the resulting
                            lookup table.
        """
        if channels not in (None, 3, 4):
            msg = "Only 3 or 4 output channels are supported"
            raise ValueError(msg)
        ch_in = self.channels
        ch_out = channels or ch_in
        size_1d, size_2d, size_3d = self.size

        table = [0] * (size_1d * size_2d * size_3d * ch_out)
        idx_in = 0
        idx_out = 0
        for b in range(size_3d):
            for g in range(size_2d):
                for r in range(size_1d):
                    values = self.table[idx_in : idx_in + ch_in]
                    if with_normals:
                        values = callback(
                            r / (size_1d - 1),
                            g / (size_2d - 1),
                            b / (size_3d - 1),
                            *values,
                        )
                    else:
                        values = callback(*values)
                    table[idx_out : idx_out + ch_out] = values
                    idx_in += ch_in
                    idx_out += ch_out

        return type(self)(
            self.size,
            table,
            channels=ch_out,
            target_mode=target_mode or self.mode,
            _copy_table=False,
        )

    def __repr__(self):
        r = [
            f"{self.__class__.__name__} from {self.table.__class__.__name__}",
            "size={:d}x{:d}x{:d}".format(*self.size),
            f"channels={self.channels:d}",
        ]
        if self.mode:
            r.append(f"target_mode={self.mode}")
        return "<{}>".format(" ".join(r))

    def filter(self, image):
        from . import Image

        return image.color_lut_3d(
            self.mode or image.mode,
            Image.Resampling.BILINEAR,
            self.channels,
            self.size[0],
            self.size[1],
            self.size[2],
            self.table,
        )
