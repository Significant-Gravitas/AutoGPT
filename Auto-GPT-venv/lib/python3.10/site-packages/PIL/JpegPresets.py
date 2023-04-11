"""
JPEG quality settings equivalent to the Photoshop settings.
Can be used when saving JPEG files.

The following presets are available by default:
``web_low``, ``web_medium``, ``web_high``, ``web_very_high``, ``web_maximum``,
``low``, ``medium``, ``high``, ``maximum``.
More presets can be added to the :py:data:`presets` dict if needed.

To apply the preset, specify::

  quality="preset_name"

To apply only the quantization table::

  qtables="preset_name"

To apply only the subsampling setting::

  subsampling="preset_name"

Example::

  im.save("image_name.jpg", quality="web_high")

Subsampling
-----------

Subsampling is the practice of encoding images by implementing less resolution
for chroma information than for luma information.
(ref.: https://en.wikipedia.org/wiki/Chroma_subsampling)

Possible subsampling values are 0, 1 and 2 that correspond to 4:4:4, 4:2:2 and
4:2:0.

You can get the subsampling of a JPEG with the
:func:`.JpegImagePlugin.get_sampling` function.

In JPEG compressed data a JPEG marker is used instead of an EXIF tag.
(ref.: https://exiv2.org/tags.html)


Quantization tables
-------------------

They are values use by the DCT (Discrete cosine transform) to remove
*unnecessary* information from the image (the lossy part of the compression).
(ref.: https://en.wikipedia.org/wiki/Quantization_matrix#Quantization_matrices,
https://en.wikipedia.org/wiki/JPEG#Quantization)

You can get the quantization tables of a JPEG with::

  im.quantization

This will return a dict with a number of lists. You can pass this dict
directly as the qtables argument when saving a JPEG.

The quantization table format in presets is a list with sublists. These formats
are interchangeable.

Libjpeg ref.:
https://web.archive.org/web/20120328125543/http://www.jpegcameras.com/libjpeg/libjpeg-3.html

"""

# fmt: off
presets = {
            'web_low':      {'subsampling':  2,  # "4:2:0"
                             'quantization': [
                               [20, 16, 25, 39, 50, 46, 62, 68,
                                16, 18, 23, 38, 38, 53, 65, 68,
                                25, 23, 31, 38, 53, 65, 68, 68,
                                39, 38, 38, 53, 65, 68, 68, 68,
                                50, 38, 53, 65, 68, 68, 68, 68,
                                46, 53, 65, 68, 68, 68, 68, 68,
                                62, 65, 68, 68, 68, 68, 68, 68,
                                68, 68, 68, 68, 68, 68, 68, 68],
                               [21, 25, 32, 38, 54, 68, 68, 68,
                                25, 28, 24, 38, 54, 68, 68, 68,
                                32, 24, 32, 43, 66, 68, 68, 68,
                                38, 38, 43, 53, 68, 68, 68, 68,
                                54, 54, 66, 68, 68, 68, 68, 68,
                                68, 68, 68, 68, 68, 68, 68, 68,
                                68, 68, 68, 68, 68, 68, 68, 68,
                                68, 68, 68, 68, 68, 68, 68, 68]
                              ]},
            'web_medium':   {'subsampling':  2,  # "4:2:0"
                             'quantization': [
                               [16, 11, 11, 16, 23, 27, 31, 30,
                                11, 12, 12, 15, 20, 23, 23, 30,
                                11, 12, 13, 16, 23, 26, 35, 47,
                                16, 15, 16, 23, 26, 37, 47, 64,
                                23, 20, 23, 26, 39, 51, 64, 64,
                                27, 23, 26, 37, 51, 64, 64, 64,
                                31, 23, 35, 47, 64, 64, 64, 64,
                                30, 30, 47, 64, 64, 64, 64, 64],
                               [17, 15, 17, 21, 20, 26, 38, 48,
                                15, 19, 18, 17, 20, 26, 35, 43,
                                17, 18, 20, 22, 26, 30, 46, 53,
                                21, 17, 22, 28, 30, 39, 53, 64,
                                20, 20, 26, 30, 39, 48, 64, 64,
                                26, 26, 30, 39, 48, 63, 64, 64,
                                38, 35, 46, 53, 64, 64, 64, 64,
                                48, 43, 53, 64, 64, 64, 64, 64]
                             ]},
            'web_high':     {'subsampling':  0,  # "4:4:4"
                             'quantization': [
                               [6,   4,  4,  6,  9, 11, 12, 16,
                                4,   5,  5,  6,  8, 10, 12, 12,
                                4,   5,  5,  6, 10, 12, 14, 19,
                                6,   6,  6, 11, 12, 15, 19, 28,
                                9,   8, 10, 12, 16, 20, 27, 31,
                                11, 10, 12, 15, 20, 27, 31, 31,
                                12, 12, 14, 19, 27, 31, 31, 31,
                                16, 12, 19, 28, 31, 31, 31, 31],
                               [7,   7, 13, 24, 26, 31, 31, 31,
                                7,  12, 16, 21, 31, 31, 31, 31,
                                13, 16, 17, 31, 31, 31, 31, 31,
                                24, 21, 31, 31, 31, 31, 31, 31,
                                26, 31, 31, 31, 31, 31, 31, 31,
                                31, 31, 31, 31, 31, 31, 31, 31,
                                31, 31, 31, 31, 31, 31, 31, 31,
                                31, 31, 31, 31, 31, 31, 31, 31]
                             ]},
            'web_very_high': {'subsampling':  0,  # "4:4:4"
                              'quantization': [
                               [2,   2,  2,  2,  3,  4,  5,  6,
                                2,   2,  2,  2,  3,  4,  5,  6,
                                2,   2,  2,  2,  4,  5,  7,  9,
                                2,   2,  2,  4,  5,  7,  9, 12,
                                3,   3,  4,  5,  8, 10, 12, 12,
                                4,   4,  5,  7, 10, 12, 12, 12,
                                5,   5,  7,  9, 12, 12, 12, 12,
                                6,   6,  9, 12, 12, 12, 12, 12],
                               [3,   3,  5,  9, 13, 15, 15, 15,
                                3,   4,  6, 11, 14, 12, 12, 12,
                                5,   6,  9, 14, 12, 12, 12, 12,
                                9,  11, 14, 12, 12, 12, 12, 12,
                                13, 14, 12, 12, 12, 12, 12, 12,
                                15, 12, 12, 12, 12, 12, 12, 12,
                                15, 12, 12, 12, 12, 12, 12, 12,
                                15, 12, 12, 12, 12, 12, 12, 12]
                              ]},
            'web_maximum':  {'subsampling':  0,  # "4:4:4"
                             'quantization': [
                                [1,  1,  1,  1,  1,  1,  1,  1,
                                 1,  1,  1,  1,  1,  1,  1,  1,
                                 1,  1,  1,  1,  1,  1,  1,  2,
                                 1,  1,  1,  1,  1,  1,  2,  2,
                                 1,  1,  1,  1,  1,  2,  2,  3,
                                 1,  1,  1,  1,  2,  2,  3,  3,
                                 1,  1,  1,  2,  2,  3,  3,  3,
                                 1,  1,  2,  2,  3,  3,  3,  3],
                                [1,  1,  1,  2,  2,  3,  3,  3,
                                 1,  1,  1,  2,  3,  3,  3,  3,
                                 1,  1,  1,  3,  3,  3,  3,  3,
                                 2,  2,  3,  3,  3,  3,  3,  3,
                                 2,  3,  3,  3,  3,  3,  3,  3,
                                 3,  3,  3,  3,  3,  3,  3,  3,
                                 3,  3,  3,  3,  3,  3,  3,  3,
                                 3,  3,  3,  3,  3,  3,  3,  3]
                             ]},
            'low':          {'subsampling':  2,  # "4:2:0"
                             'quantization': [
                               [18, 14, 14, 21, 30, 35, 34, 17,
                                14, 16, 16, 19, 26, 23, 12, 12,
                                14, 16, 17, 21, 23, 12, 12, 12,
                                21, 19, 21, 23, 12, 12, 12, 12,
                                30, 26, 23, 12, 12, 12, 12, 12,
                                35, 23, 12, 12, 12, 12, 12, 12,
                                34, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12],
                               [20, 19, 22, 27, 20, 20, 17, 17,
                                19, 25, 23, 14, 14, 12, 12, 12,
                                22, 23, 14, 14, 12, 12, 12, 12,
                                27, 14, 14, 12, 12, 12, 12, 12,
                                20, 14, 12, 12, 12, 12, 12, 12,
                                20, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12]
                             ]},
            'medium':       {'subsampling':  2,  # "4:2:0"
                             'quantization': [
                               [12,  8,  8, 12, 17, 21, 24, 17,
                                8,   9,  9, 11, 15, 19, 12, 12,
                                8,   9, 10, 12, 19, 12, 12, 12,
                                12, 11, 12, 21, 12, 12, 12, 12,
                                17, 15, 19, 12, 12, 12, 12, 12,
                                21, 19, 12, 12, 12, 12, 12, 12,
                                24, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12],
                               [13, 11, 13, 16, 20, 20, 17, 17,
                                11, 14, 14, 14, 14, 12, 12, 12,
                                13, 14, 14, 14, 12, 12, 12, 12,
                                16, 14, 14, 12, 12, 12, 12, 12,
                                20, 14, 12, 12, 12, 12, 12, 12,
                                20, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12]
                             ]},
            'high':         {'subsampling':  0,  # "4:4:4"
                             'quantization': [
                               [6,   4,  4,  6,  9, 11, 12, 16,
                                4,   5,  5,  6,  8, 10, 12, 12,
                                4,   5,  5,  6, 10, 12, 12, 12,
                                6,   6,  6, 11, 12, 12, 12, 12,
                                9,   8, 10, 12, 12, 12, 12, 12,
                                11, 10, 12, 12, 12, 12, 12, 12,
                                12, 12, 12, 12, 12, 12, 12, 12,
                                16, 12, 12, 12, 12, 12, 12, 12],
                               [7,   7, 13, 24, 20, 20, 17, 17,
                                7,  12, 16, 14, 14, 12, 12, 12,
                                13, 16, 14, 14, 12, 12, 12, 12,
                                24, 14, 14, 12, 12, 12, 12, 12,
                                20, 14, 12, 12, 12, 12, 12, 12,
                                20, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12,
                                17, 12, 12, 12, 12, 12, 12, 12]
                             ]},
            'maximum':      {'subsampling':  0,  # "4:4:4"
                             'quantization': [
                               [2,   2,  2,  2,  3,  4,  5,  6,
                                2,   2,  2,  2,  3,  4,  5,  6,
                                2,   2,  2,  2,  4,  5,  7,  9,
                                2,   2,  2,  4,  5,  7,  9, 12,
                                3,   3,  4,  5,  8, 10, 12, 12,
                                4,   4,  5,  7, 10, 12, 12, 12,
                                5,   5,  7,  9, 12, 12, 12, 12,
                                6,   6,  9, 12, 12, 12, 12, 12],
                               [3,   3,  5,  9, 13, 15, 15, 15,
                                3,   4,  6, 10, 14, 12, 12, 12,
                                5,   6,  9, 14, 12, 12, 12, 12,
                                9,  10, 14, 12, 12, 12, 12, 12,
                                13, 14, 12, 12, 12, 12, 12, 12,
                                15, 12, 12, 12, 12, 12, 12, 12,
                                15, 12, 12, 12, 12, 12, 12, 12,
                                15, 12, 12, 12, 12, 12, 12, 12]
                             ]},
}
# fmt: on
