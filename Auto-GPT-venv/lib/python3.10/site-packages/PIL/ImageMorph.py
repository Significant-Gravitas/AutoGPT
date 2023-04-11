# A binary morphology add-on for the Python Imaging Library
#
# History:
#   2014-06-04 Initial version.
#
# Copyright (c) 2014 Dov Grobgeld <dov.grobgeld@gmail.com>

import re

from . import Image, _imagingmorph

LUT_SIZE = 1 << 9

# fmt: off
ROTATION_MATRIX = [
    6, 3, 0,
    7, 4, 1,
    8, 5, 2,
]
MIRROR_MATRIX = [
    2, 1, 0,
    5, 4, 3,
    8, 7, 6,
]
# fmt: on


class LutBuilder:
    """A class for building a MorphLut from a descriptive language

    The input patterns is a list of a strings sequences like these::

        4:(...
           .1.
           111)->1

    (whitespaces including linebreaks are ignored). The option 4
    describes a series of symmetry operations (in this case a
    4-rotation), the pattern is described by:

    - . or X - Ignore
    - 1 - Pixel is on
    - 0 - Pixel is off

    The result of the operation is described after "->" string.

    The default is to return the current pixel value, which is
    returned if no other match is found.

    Operations:

    - 4 - 4 way rotation
    - N - Negate
    - 1 - Dummy op for no other operation (an op must always be given)
    - M - Mirroring

    Example::

        lb = LutBuilder(patterns = ["4:(... .1. 111)->1"])
        lut = lb.build_lut()

    """

    def __init__(self, patterns=None, op_name=None):
        if patterns is not None:
            self.patterns = patterns
        else:
            self.patterns = []
        self.lut = None
        if op_name is not None:
            known_patterns = {
                "corner": ["1:(... ... ...)->0", "4:(00. 01. ...)->1"],
                "dilation4": ["4:(... .0. .1.)->1"],
                "dilation8": ["4:(... .0. .1.)->1", "4:(... .0. ..1)->1"],
                "erosion4": ["4:(... .1. .0.)->0"],
                "erosion8": ["4:(... .1. .0.)->0", "4:(... .1. ..0)->0"],
                "edge": [
                    "1:(... ... ...)->0",
                    "4:(.0. .1. ...)->1",
                    "4:(01. .1. ...)->1",
                ],
            }
            if op_name not in known_patterns:
                msg = "Unknown pattern " + op_name + "!"
                raise Exception(msg)

            self.patterns = known_patterns[op_name]

    def add_patterns(self, patterns):
        self.patterns += patterns

    def build_default_lut(self):
        symbols = [0, 1]
        m = 1 << 4  # pos of current pixel
        self.lut = bytearray(symbols[(i & m) > 0] for i in range(LUT_SIZE))

    def get_lut(self):
        return self.lut

    def _string_permute(self, pattern, permutation):
        """string_permute takes a pattern and a permutation and returns the
        string permuted according to the permutation list.
        """
        assert len(permutation) == 9
        return "".join(pattern[p] for p in permutation)

    def _pattern_permute(self, basic_pattern, options, basic_result):
        """pattern_permute takes a basic pattern and its result and clones
        the pattern according to the modifications described in the $options
        parameter. It returns a list of all cloned patterns."""
        patterns = [(basic_pattern, basic_result)]

        # rotations
        if "4" in options:
            res = patterns[-1][1]
            for i in range(4):
                patterns.append(
                    (self._string_permute(patterns[-1][0], ROTATION_MATRIX), res)
                )
        # mirror
        if "M" in options:
            n = len(patterns)
            for pattern, res in patterns[:n]:
                patterns.append((self._string_permute(pattern, MIRROR_MATRIX), res))

        # negate
        if "N" in options:
            n = len(patterns)
            for pattern, res in patterns[:n]:
                # Swap 0 and 1
                pattern = pattern.replace("0", "Z").replace("1", "0").replace("Z", "1")
                res = 1 - int(res)
                patterns.append((pattern, res))

        return patterns

    def build_lut(self):
        """Compile all patterns into a morphology lut.

        TBD :Build based on (file) morphlut:modify_lut
        """
        self.build_default_lut()
        patterns = []

        # Parse and create symmetries of the patterns strings
        for p in self.patterns:
            m = re.search(r"(\w*):?\s*\((.+?)\)\s*->\s*(\d)", p.replace("\n", ""))
            if not m:
                msg = 'Syntax error in pattern "' + p + '"'
                raise Exception(msg)
            options = m.group(1)
            pattern = m.group(2)
            result = int(m.group(3))

            # Get rid of spaces
            pattern = pattern.replace(" ", "").replace("\n", "")

            patterns += self._pattern_permute(pattern, options, result)

        # compile the patterns into regular expressions for speed
        for i, pattern in enumerate(patterns):
            p = pattern[0].replace(".", "X").replace("X", "[01]")
            p = re.compile(p)
            patterns[i] = (p, pattern[1])

        # Step through table and find patterns that match.
        # Note that all the patterns are searched. The last one
        # caught overrides
        for i in range(LUT_SIZE):
            # Build the bit pattern
            bitpattern = bin(i)[2:]
            bitpattern = ("0" * (9 - len(bitpattern)) + bitpattern)[::-1]

            for p, r in patterns:
                if p.match(bitpattern):
                    self.lut[i] = [0, 1][r]

        return self.lut


class MorphOp:
    """A class for binary morphological operators"""

    def __init__(self, lut=None, op_name=None, patterns=None):
        """Create a binary morphological operator"""
        self.lut = lut
        if op_name is not None:
            self.lut = LutBuilder(op_name=op_name).build_lut()
        elif patterns is not None:
            self.lut = LutBuilder(patterns=patterns).build_lut()

    def apply(self, image):
        """Run a single morphological operation on an image

        Returns a tuple of the number of changed pixels and the
        morphed image"""
        if self.lut is None:
            msg = "No operator loaded"
            raise Exception(msg)

        if image.mode != "L":
            msg = "Image mode must be L"
            raise ValueError(msg)
        outimage = Image.new(image.mode, image.size, None)
        count = _imagingmorph.apply(bytes(self.lut), image.im.id, outimage.im.id)
        return count, outimage

    def match(self, image):
        """Get a list of coordinates matching the morphological operation on
        an image.

        Returns a list of tuples of (x,y) coordinates
        of all matching pixels. See :ref:`coordinate-system`."""
        if self.lut is None:
            msg = "No operator loaded"
            raise Exception(msg)

        if image.mode != "L":
            msg = "Image mode must be L"
            raise ValueError(msg)
        return _imagingmorph.match(bytes(self.lut), image.im.id)

    def get_on_pixels(self, image):
        """Get a list of all turned on pixels in a binary image

        Returns a list of tuples of (x,y) coordinates
        of all matching pixels. See :ref:`coordinate-system`."""

        if image.mode != "L":
            msg = "Image mode must be L"
            raise ValueError(msg)
        return _imagingmorph.get_on_pixels(image.im.id)

    def load_lut(self, filename):
        """Load an operator from an mrl file"""
        with open(filename, "rb") as f:
            self.lut = bytearray(f.read())

        if len(self.lut) != LUT_SIZE:
            self.lut = None
            msg = "Wrong size operator file!"
            raise Exception(msg)

    def save_lut(self, filename):
        """Save an operator to an mrl file"""
        if self.lut is None:
            msg = "No operator loaded"
            raise Exception(msg)
        with open(filename, "wb") as f:
            f.write(self.lut)

    def set_lut(self, lut):
        """Set the lut from an external source"""
        self.lut = lut
