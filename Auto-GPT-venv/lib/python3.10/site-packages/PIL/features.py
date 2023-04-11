import collections
import os
import sys
import warnings

import PIL

from . import Image

modules = {
    "pil": ("PIL._imaging", "PILLOW_VERSION"),
    "tkinter": ("PIL._tkinter_finder", "tk_version"),
    "freetype2": ("PIL._imagingft", "freetype2_version"),
    "littlecms2": ("PIL._imagingcms", "littlecms_version"),
    "webp": ("PIL._webp", "webpdecoder_version"),
}


def check_module(feature):
    """
    Checks if a module is available.

    :param feature: The module to check for.
    :returns: ``True`` if available, ``False`` otherwise.
    :raises ValueError: If the module is not defined in this version of Pillow.
    """
    if not (feature in modules):
        msg = f"Unknown module {feature}"
        raise ValueError(msg)

    module, ver = modules[feature]

    try:
        __import__(module)
        return True
    except ModuleNotFoundError:
        return False
    except ImportError as ex:
        warnings.warn(str(ex))
        return False


def version_module(feature):
    """
    :param feature: The module to check for.
    :returns:
        The loaded version number as a string, or ``None`` if unknown or not available.
    :raises ValueError: If the module is not defined in this version of Pillow.
    """
    if not check_module(feature):
        return None

    module, ver = modules[feature]

    if ver is None:
        return None

    return getattr(__import__(module, fromlist=[ver]), ver)


def get_supported_modules():
    """
    :returns: A list of all supported modules.
    """
    return [f for f in modules if check_module(f)]


codecs = {
    "jpg": ("jpeg", "jpeglib"),
    "jpg_2000": ("jpeg2k", "jp2klib"),
    "zlib": ("zip", "zlib"),
    "libtiff": ("libtiff", "libtiff"),
}


def check_codec(feature):
    """
    Checks if a codec is available.

    :param feature: The codec to check for.
    :returns: ``True`` if available, ``False`` otherwise.
    :raises ValueError: If the codec is not defined in this version of Pillow.
    """
    if feature not in codecs:
        msg = f"Unknown codec {feature}"
        raise ValueError(msg)

    codec, lib = codecs[feature]

    return codec + "_encoder" in dir(Image.core)


def version_codec(feature):
    """
    :param feature: The codec to check for.
    :returns:
        The version number as a string, or ``None`` if not available.
        Checked at compile time for ``jpg``, run-time otherwise.
    :raises ValueError: If the codec is not defined in this version of Pillow.
    """
    if not check_codec(feature):
        return None

    codec, lib = codecs[feature]

    version = getattr(Image.core, lib + "_version")

    if feature == "libtiff":
        return version.split("\n")[0].split("Version ")[1]

    return version


def get_supported_codecs():
    """
    :returns: A list of all supported codecs.
    """
    return [f for f in codecs if check_codec(f)]


features = {
    "webp_anim": ("PIL._webp", "HAVE_WEBPANIM", None),
    "webp_mux": ("PIL._webp", "HAVE_WEBPMUX", None),
    "transp_webp": ("PIL._webp", "HAVE_TRANSPARENCY", None),
    "raqm": ("PIL._imagingft", "HAVE_RAQM", "raqm_version"),
    "fribidi": ("PIL._imagingft", "HAVE_FRIBIDI", "fribidi_version"),
    "harfbuzz": ("PIL._imagingft", "HAVE_HARFBUZZ", "harfbuzz_version"),
    "libjpeg_turbo": ("PIL._imaging", "HAVE_LIBJPEGTURBO", "libjpeg_turbo_version"),
    "libimagequant": ("PIL._imaging", "HAVE_LIBIMAGEQUANT", "imagequant_version"),
    "xcb": ("PIL._imaging", "HAVE_XCB", None),
}


def check_feature(feature):
    """
    Checks if a feature is available.

    :param feature: The feature to check for.
    :returns: ``True`` if available, ``False`` if unavailable, ``None`` if unknown.
    :raises ValueError: If the feature is not defined in this version of Pillow.
    """
    if feature not in features:
        msg = f"Unknown feature {feature}"
        raise ValueError(msg)

    module, flag, ver = features[feature]

    try:
        imported_module = __import__(module, fromlist=["PIL"])
        return getattr(imported_module, flag)
    except ModuleNotFoundError:
        return None
    except ImportError as ex:
        warnings.warn(str(ex))
        return None


def version_feature(feature):
    """
    :param feature: The feature to check for.
    :returns: The version number as a string, or ``None`` if not available.
    :raises ValueError: If the feature is not defined in this version of Pillow.
    """
    if not check_feature(feature):
        return None

    module, flag, ver = features[feature]

    if ver is None:
        return None

    return getattr(__import__(module, fromlist=[ver]), ver)


def get_supported_features():
    """
    :returns: A list of all supported features.
    """
    return [f for f in features if check_feature(f)]


def check(feature):
    """
    :param feature: A module, codec, or feature name.
    :returns:
        ``True`` if the module, codec, or feature is available,
        ``False`` or ``None`` otherwise.
    """

    if feature in modules:
        return check_module(feature)
    if feature in codecs:
        return check_codec(feature)
    if feature in features:
        return check_feature(feature)
    warnings.warn(f"Unknown feature '{feature}'.", stacklevel=2)
    return False


def version(feature):
    """
    :param feature:
        The module, codec, or feature to check for.
    :returns:
        The version number as a string, or ``None`` if unknown or not available.
    """
    if feature in modules:
        return version_module(feature)
    if feature in codecs:
        return version_codec(feature)
    if feature in features:
        return version_feature(feature)
    return None


def get_supported():
    """
    :returns: A list of all supported modules, features, and codecs.
    """

    ret = get_supported_modules()
    ret.extend(get_supported_features())
    ret.extend(get_supported_codecs())
    return ret


def pilinfo(out=None, supported_formats=True):
    """
    Prints information about this installation of Pillow.
    This function can be called with ``python3 -m PIL``.

    :param out:
        The output stream to print to. Defaults to ``sys.stdout`` if ``None``.
    :param supported_formats:
        If ``True``, a list of all supported image file formats will be printed.
    """

    if out is None:
        out = sys.stdout

    Image.init()

    print("-" * 68, file=out)
    print(f"Pillow {PIL.__version__}", file=out)
    py_version = sys.version.splitlines()
    print(f"Python {py_version[0].strip()}", file=out)
    for py_version in py_version[1:]:
        print(f"       {py_version.strip()}", file=out)
    print("-" * 68, file=out)
    print(
        f"Python modules loaded from {os.path.dirname(Image.__file__)}",
        file=out,
    )
    print(
        f"Binary modules loaded from {os.path.dirname(Image.core.__file__)}",
        file=out,
    )
    print("-" * 68, file=out)

    for name, feature in [
        ("pil", "PIL CORE"),
        ("tkinter", "TKINTER"),
        ("freetype2", "FREETYPE2"),
        ("littlecms2", "LITTLECMS2"),
        ("webp", "WEBP"),
        ("transp_webp", "WEBP Transparency"),
        ("webp_mux", "WEBPMUX"),
        ("webp_anim", "WEBP Animation"),
        ("jpg", "JPEG"),
        ("jpg_2000", "OPENJPEG (JPEG2000)"),
        ("zlib", "ZLIB (PNG/ZIP)"),
        ("libtiff", "LIBTIFF"),
        ("raqm", "RAQM (Bidirectional Text)"),
        ("libimagequant", "LIBIMAGEQUANT (Quantization method)"),
        ("xcb", "XCB (X protocol)"),
    ]:
        if check(name):
            if name == "jpg" and check_feature("libjpeg_turbo"):
                v = "libjpeg-turbo " + version_feature("libjpeg_turbo")
            else:
                v = version(name)
            if v is not None:
                version_static = name in ("pil", "jpg")
                if name == "littlecms2":
                    # this check is also in src/_imagingcms.c:setup_module()
                    version_static = tuple(int(x) for x in v.split(".")) < (2, 7)
                t = "compiled for" if version_static else "loaded"
                if name == "raqm":
                    for f in ("fribidi", "harfbuzz"):
                        v2 = version_feature(f)
                        if v2 is not None:
                            v += f", {f} {v2}"
                print("---", feature, "support ok,", t, v, file=out)
            else:
                print("---", feature, "support ok", file=out)
        else:
            print("***", feature, "support not installed", file=out)
    print("-" * 68, file=out)

    if supported_formats:
        extensions = collections.defaultdict(list)
        for ext, i in Image.EXTENSION.items():
            extensions[i].append(ext)

        for i in sorted(Image.ID):
            line = f"{i}"
            if i in Image.MIME:
                line = f"{line} {Image.MIME[i]}"
            print(line, file=out)

            if i in extensions:
                print(
                    "Extensions: {}".format(", ".join(sorted(extensions[i]))), file=out
                )

            features = []
            if i in Image.OPEN:
                features.append("open")
            if i in Image.SAVE:
                features.append("save")
            if i in Image.SAVE_ALL:
                features.append("save_all")
            if i in Image.DECODERS:
                features.append("decode")
            if i in Image.ENCODERS:
                features.append("encode")

            print("Features: {}".format(", ".join(features)), file=out)
            print("-" * 68, file=out)
