#
# Python Imaging Library
# $Id$
#
# stuff to read (and render) GIMP gradient files
#
# History:
#       97-08-23 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#

"""
Stuff to translate curve segments to palette values (derived from
the corresponding code in GIMP, written by Federico Mena Quintero.
See the GIMP distribution for more information.)
"""


from math import log, pi, sin, sqrt

from ._binary import o8

EPSILON = 1e-10
""""""  # Enable auto-doc for data member


def linear(middle, pos):
    if pos <= middle:
        if middle < EPSILON:
            return 0.0
        else:
            return 0.5 * pos / middle
    else:
        pos = pos - middle
        middle = 1.0 - middle
        if middle < EPSILON:
            return 1.0
        else:
            return 0.5 + 0.5 * pos / middle


def curved(middle, pos):
    return pos ** (log(0.5) / log(max(middle, EPSILON)))


def sine(middle, pos):
    return (sin((-pi / 2.0) + pi * linear(middle, pos)) + 1.0) / 2.0


def sphere_increasing(middle, pos):
    return sqrt(1.0 - (linear(middle, pos) - 1.0) ** 2)


def sphere_decreasing(middle, pos):
    return 1.0 - sqrt(1.0 - linear(middle, pos) ** 2)


SEGMENTS = [linear, curved, sine, sphere_increasing, sphere_decreasing]
""""""  # Enable auto-doc for data member


class GradientFile:
    gradient = None

    def getpalette(self, entries=256):
        palette = []

        ix = 0
        x0, x1, xm, rgb0, rgb1, segment = self.gradient[ix]

        for i in range(entries):
            x = i / (entries - 1)

            while x1 < x:
                ix += 1
                x0, x1, xm, rgb0, rgb1, segment = self.gradient[ix]

            w = x1 - x0

            if w < EPSILON:
                scale = segment(0.5, 0.5)
            else:
                scale = segment((xm - x0) / w, (x - x0) / w)

            # expand to RGBA
            r = o8(int(255 * ((rgb1[0] - rgb0[0]) * scale + rgb0[0]) + 0.5))
            g = o8(int(255 * ((rgb1[1] - rgb0[1]) * scale + rgb0[1]) + 0.5))
            b = o8(int(255 * ((rgb1[2] - rgb0[2]) * scale + rgb0[2]) + 0.5))
            a = o8(int(255 * ((rgb1[3] - rgb0[3]) * scale + rgb0[3]) + 0.5))

            # add to palette
            palette.append(r + g + b + a)

        return b"".join(palette), "RGBA"


class GimpGradientFile(GradientFile):
    """File handler for GIMP's gradient format."""

    def __init__(self, fp):
        if fp.readline()[:13] != b"GIMP Gradient":
            msg = "not a GIMP gradient file"
            raise SyntaxError(msg)

        line = fp.readline()

        # GIMP 1.2 gradient files don't contain a name, but GIMP 1.3 files do
        if line.startswith(b"Name: "):
            line = fp.readline().strip()

        count = int(line)

        gradient = []

        for i in range(count):
            s = fp.readline().split()
            w = [float(x) for x in s[:11]]

            x0, x1 = w[0], w[2]
            xm = w[1]
            rgb0 = w[3:7]
            rgb1 = w[7:11]

            segment = SEGMENTS[int(s[11])]
            cspace = int(s[12])

            if cspace != 0:
                msg = "cannot handle HSV colour space"
                raise OSError(msg)

            gradient.append((x0, x1, xm, rgb0, rgb1, segment))

        self.gradient = gradient
