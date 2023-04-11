#
# The Python Imaging Library
# $Id$
#
# bitmap distribution font (bdf) file parser
#
# history:
# 1996-05-16 fl   created (as bdf2pil)
# 1997-08-25 fl   converted to FontFile driver
# 2001-05-25 fl   removed bogus __init__ call
# 2002-11-20 fl   robustification (from Kevin Cazabon, Dmitry Vasiliev)
# 2003-04-22 fl   more robustification (from Graham Dumpleton)
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1997-2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

"""
Parse X Bitmap Distribution Format (BDF)
"""


from . import FontFile, Image

bdf_slant = {
    "R": "Roman",
    "I": "Italic",
    "O": "Oblique",
    "RI": "Reverse Italic",
    "RO": "Reverse Oblique",
    "OT": "Other",
}

bdf_spacing = {"P": "Proportional", "M": "Monospaced", "C": "Cell"}


def bdf_char(f):
    # skip to STARTCHAR
    while True:
        s = f.readline()
        if not s:
            return None
        if s[:9] == b"STARTCHAR":
            break
    id = s[9:].strip().decode("ascii")

    # load symbol properties
    props = {}
    while True:
        s = f.readline()
        if not s or s[:6] == b"BITMAP":
            break
        i = s.find(b" ")
        props[s[:i].decode("ascii")] = s[i + 1 : -1].decode("ascii")

    # load bitmap
    bitmap = []
    while True:
        s = f.readline()
        if not s or s[:7] == b"ENDCHAR":
            break
        bitmap.append(s[:-1])
    bitmap = b"".join(bitmap)

    # The word BBX
    # followed by the width in x (BBw), height in y (BBh),
    # and x and y displacement (BBxoff0, BByoff0)
    # of the lower left corner from the origin of the character.
    width, height, x_disp, y_disp = [int(p) for p in props["BBX"].split()]

    # The word DWIDTH
    # followed by the width in x and y of the character in device pixels.
    dwx, dwy = [int(p) for p in props["DWIDTH"].split()]

    bbox = (
        (dwx, dwy),
        (x_disp, -y_disp - height, width + x_disp, -y_disp),
        (0, 0, width, height),
    )

    try:
        im = Image.frombytes("1", (width, height), bitmap, "hex", "1")
    except ValueError:
        # deal with zero-width characters
        im = Image.new("1", (width, height))

    return id, int(props["ENCODING"]), bbox, im


class BdfFontFile(FontFile.FontFile):
    """Font file plugin for the X11 BDF format."""

    def __init__(self, fp):
        super().__init__()

        s = fp.readline()
        if s[:13] != b"STARTFONT 2.1":
            msg = "not a valid BDF file"
            raise SyntaxError(msg)

        props = {}
        comments = []

        while True:
            s = fp.readline()
            if not s or s[:13] == b"ENDPROPERTIES":
                break
            i = s.find(b" ")
            props[s[:i].decode("ascii")] = s[i + 1 : -1].decode("ascii")
            if s[:i] in [b"COMMENT", b"COPYRIGHT"]:
                if s.find(b"LogicalFontDescription") < 0:
                    comments.append(s[i + 1 : -1].decode("ascii"))

        while True:
            c = bdf_char(fp)
            if not c:
                break
            id, ch, (xy, dst, src), im = c
            if 0 <= ch < len(self.glyph):
                self.glyph[ch] = xy, dst, src, im
