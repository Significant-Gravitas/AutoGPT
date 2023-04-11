#
# The Python Imaging Library.
# $Id$
#
# im.show() drivers
#
# History:
# 2008-04-06 fl   Created
#
# Copyright (c) Secret Labs AB 2008.
#
# See the README file for information on usage and redistribution.
#
import os
import shutil
import subprocess
import sys
from shlex import quote

from PIL import Image

from ._deprecate import deprecate

_viewers = []


def register(viewer, order=1):
    """
    The :py:func:`register` function is used to register additional viewers::

        from PIL import ImageShow
        ImageShow.register(MyViewer())  # MyViewer will be used as a last resort
        ImageShow.register(MySecondViewer(), 0)  # MySecondViewer will be prioritised
        ImageShow.register(ImageShow.XVViewer(), 0)  # XVViewer will be prioritised

    :param viewer: The viewer to be registered.
    :param order:
        Zero or a negative integer to prepend this viewer to the list,
        a positive integer to append it.
    """
    try:
        if issubclass(viewer, Viewer):
            viewer = viewer()
    except TypeError:
        pass  # raised if viewer wasn't a class
    if order > 0:
        _viewers.append(viewer)
    else:
        _viewers.insert(0, viewer)


def show(image, title=None, **options):
    r"""
    Display a given image.

    :param image: An image object.
    :param title: Optional title. Not all viewers can display the title.
    :param \**options: Additional viewer options.
    :returns: ``True`` if a suitable viewer was found, ``False`` otherwise.
    """
    for viewer in _viewers:
        if viewer.show(image, title=title, **options):
            return True
    return False


class Viewer:
    """Base class for viewers."""

    # main api

    def show(self, image, **options):
        """
        The main function for displaying an image.
        Converts the given image to the target format and displays it.
        """

        if not (
            image.mode in ("1", "RGBA")
            or (self.format == "PNG" and image.mode in ("I;16", "LA"))
        ):
            base = Image.getmodebase(image.mode)
            if image.mode != base:
                image = image.convert(base)

        return self.show_image(image, **options)

    # hook methods

    format = None
    """The format to convert the image into."""
    options = {}
    """Additional options used to convert the image."""

    def get_format(self, image):
        """Return format name, or ``None`` to save as PGM/PPM."""
        return self.format

    def get_command(self, file, **options):
        """
        Returns the command used to display the file.
        Not implemented in the base class.
        """
        raise NotImplementedError

    def save_image(self, image):
        """Save to temporary file and return filename."""
        return image._dump(format=self.get_format(image), **self.options)

    def show_image(self, image, **options):
        """Display the given image."""
        return self.show_file(self.save_image(image), **options)

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and will be removed in Pillow 10.0.0 (2023-07-01). ``path`` should be used
        instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        os.system(self.get_command(path, **options))  # nosec
        return 1


# --------------------------------------------------------------------


class WindowsViewer(Viewer):
    """The default viewer on Windows is the default system application for PNG files."""

    format = "PNG"
    options = {"compress_level": 1, "save_all": True}

    def get_command(self, file, **options):
        return (
            f'start "Pillow" /WAIT "{file}" '
            "&& ping -n 4 127.0.0.1 >NUL "
            f'&& del /f "{file}"'
        )


if sys.platform == "win32":
    register(WindowsViewer)


class MacViewer(Viewer):
    """The default viewer on macOS using ``Preview.app``."""

    format = "PNG"
    options = {"compress_level": 1, "save_all": True}

    def get_command(self, file, **options):
        # on darwin open returns immediately resulting in the temp
        # file removal while app is opening
        command = "open -a Preview.app"
        command = f"({command} {quote(file)}; sleep 20; rm -f {quote(file)})&"
        return command

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and will be removed in Pillow 10.0.0 (2023-07-01). ``path`` should be used
        instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        subprocess.call(["open", "-a", "Preview.app", path])
        executable = sys.executable or shutil.which("python3")
        if executable:
            subprocess.Popen(
                [
                    executable,
                    "-c",
                    "import os, sys, time; time.sleep(20); os.remove(sys.argv[1])",
                    path,
                ]
            )
        return 1


if sys.platform == "darwin":
    register(MacViewer)


class UnixViewer(Viewer):
    format = "PNG"
    options = {"compress_level": 1, "save_all": True}

    def get_command(self, file, **options):
        command = self.get_command_ex(file, **options)[0]
        return f"({command} {quote(file)}"


class XDGViewer(UnixViewer):
    """
    The freedesktop.org ``xdg-open`` command.
    """

    def get_command_ex(self, file, **options):
        command = executable = "xdg-open"
        return command, executable

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and will be removed in Pillow 10.0.0 (2023-07-01). ``path`` should be used
        instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        subprocess.Popen(["xdg-open", path])
        return 1


class DisplayViewer(UnixViewer):
    """
    The ImageMagick ``display`` command.
    This viewer supports the ``title`` parameter.
    """

    def get_command_ex(self, file, title=None, **options):
        command = executable = "display"
        if title:
            command += f" -title {quote(title)}"
        return command, executable

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and ``path`` should be used instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        args = ["display"]
        title = options.get("title")
        if title:
            args += ["-title", title]
        args.append(path)

        subprocess.Popen(args)
        return 1


class GmDisplayViewer(UnixViewer):
    """The GraphicsMagick ``gm display`` command."""

    def get_command_ex(self, file, **options):
        executable = "gm"
        command = "gm display"
        return command, executable

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and ``path`` should be used instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        subprocess.Popen(["gm", "display", path])
        return 1


class EogViewer(UnixViewer):
    """The GNOME Image Viewer ``eog`` command."""

    def get_command_ex(self, file, **options):
        executable = "eog"
        command = "eog -n"
        return command, executable

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and ``path`` should be used instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        subprocess.Popen(["eog", "-n", path])
        return 1


class XVViewer(UnixViewer):
    """
    The X Viewer ``xv`` command.
    This viewer supports the ``title`` parameter.
    """

    def get_command_ex(self, file, title=None, **options):
        # note: xv is pretty outdated.  most modern systems have
        # imagemagick's display command instead.
        command = executable = "xv"
        if title:
            command += f" -name {quote(title)}"
        return command, executable

    def show_file(self, path=None, **options):
        """
        Display given file.

        Before Pillow 9.1.0, the first argument was ``file``. This is now deprecated,
        and ``path`` should be used instead.
        """
        if path is None:
            if "file" in options:
                deprecate("The 'file' argument", 10, "'path'")
                path = options.pop("file")
            else:
                msg = "Missing required argument: 'path'"
                raise TypeError(msg)
        args = ["xv"]
        title = options.get("title")
        if title:
            args += ["-name", title]
        args.append(path)

        subprocess.Popen(args)
        return 1


if sys.platform not in ("win32", "darwin"):  # unixoids
    if shutil.which("xdg-open"):
        register(XDGViewer)
    if shutil.which("display"):
        register(DisplayViewer)
    if shutil.which("gm"):
        register(GmDisplayViewer)
    if shutil.which("eog"):
        register(EogViewer)
    if shutil.which("xv"):
        register(XVViewer)


class IPythonViewer(Viewer):
    """The viewer for IPython frontends."""

    def show_image(self, image, **options):
        ipython_display(image)
        return 1


try:
    from IPython.display import display as ipython_display
except ImportError:
    pass
else:
    register(IPythonViewer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Syntax: python3 ImageShow.py imagefile [title]")
        sys.exit()

    with Image.open(sys.argv[1]) as im:
        print(show(im, *sys.argv[2:]))
