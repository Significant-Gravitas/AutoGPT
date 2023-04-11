#
# The Python Imaging Library
# $Id$
#
# screen grabber
#
# History:
# 2001-04-26 fl  created
# 2001-09-17 fl  use builtin driver, if present
# 2002-11-19 fl  added grabclipboard support
#
# Copyright (c) 2001-2002 by Secret Labs AB
# Copyright (c) 2001-2002 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import os
import shutil
import subprocess
import sys
import tempfile

from . import Image


def grab(bbox=None, include_layered_windows=False, all_screens=False, xdisplay=None):
    if xdisplay is None:
        if sys.platform == "darwin":
            fh, filepath = tempfile.mkstemp(".png")
            os.close(fh)
            args = ["screencapture"]
            if bbox:
                left, top, right, bottom = bbox
                args += ["-R", f"{left},{top},{right-left},{bottom-top}"]
            subprocess.call(args + ["-x", filepath])
            im = Image.open(filepath)
            im.load()
            os.unlink(filepath)
            if bbox:
                im_resized = im.resize((right - left, bottom - top))
                im.close()
                return im_resized
            return im
        elif sys.platform == "win32":
            offset, size, data = Image.core.grabscreen_win32(
                include_layered_windows, all_screens
            )
            im = Image.frombytes(
                "RGB",
                size,
                data,
                # RGB, 32-bit line padding, origin lower left corner
                "raw",
                "BGR",
                (size[0] * 3 + 3) & -4,
                -1,
            )
            if bbox:
                x0, y0 = offset
                left, top, right, bottom = bbox
                im = im.crop((left - x0, top - y0, right - x0, bottom - y0))
            return im
        elif shutil.which("gnome-screenshot"):
            fh, filepath = tempfile.mkstemp(".png")
            os.close(fh)
            subprocess.call(["gnome-screenshot", "-f", filepath])
            im = Image.open(filepath)
            im.load()
            os.unlink(filepath)
            if bbox:
                im_cropped = im.crop(bbox)
                im.close()
                return im_cropped
            return im
    # use xdisplay=None for default display on non-win32/macOS systems
    if not Image.core.HAVE_XCB:
        msg = "Pillow was built without XCB support"
        raise OSError(msg)
    size, data = Image.core.grabscreen_x11(xdisplay)
    im = Image.frombytes("RGB", size, data, "raw", "BGRX", size[0] * 4, 1)
    if bbox:
        im = im.crop(bbox)
    return im


def grabclipboard():
    if sys.platform == "darwin":
        fh, filepath = tempfile.mkstemp(".jpg")
        os.close(fh)
        commands = [
            'set theFile to (open for access POSIX file "'
            + filepath
            + '" with write permission)',
            "try",
            "    write (the clipboard as JPEG picture) to theFile",
            "end try",
            "close access theFile",
        ]
        script = ["osascript"]
        for command in commands:
            script += ["-e", command]
        subprocess.call(script)

        im = None
        if os.stat(filepath).st_size != 0:
            im = Image.open(filepath)
            im.load()
        os.unlink(filepath)
        return im
    elif sys.platform == "win32":
        fmt, data = Image.core.grabclipboard_win32()
        if fmt == "file":  # CF_HDROP
            import struct

            o = struct.unpack_from("I", data)[0]
            if data[16] != 0:
                files = data[o:].decode("utf-16le").split("\0")
            else:
                files = data[o:].decode("mbcs").split("\0")
            return files[: files.index("")]
        if isinstance(data, bytes):
            import io

            data = io.BytesIO(data)
            if fmt == "png":
                from . import PngImagePlugin

                return PngImagePlugin.PngImageFile(data)
            elif fmt == "DIB":
                from . import BmpImagePlugin

                return BmpImagePlugin.DibImageFile(data)
        return None
    else:
        if shutil.which("wl-paste"):
            args = ["wl-paste"]
        elif shutil.which("xclip"):
            args = ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
        else:
            msg = "wl-paste or xclip is required for ImageGrab.grabclipboard() on Linux"
            raise NotImplementedError(msg)
        fh, filepath = tempfile.mkstemp()
        subprocess.call(args, stdout=fh)
        os.close(fh)
        im = Image.open(filepath)
        im.load()
        os.unlink(filepath)
        return im
