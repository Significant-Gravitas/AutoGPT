""" Stamp a Win32 binary with version information.
"""

import glob
import optparse
import os
import struct
import sys

from win32api import BeginUpdateResource, EndUpdateResource, UpdateResource

VS_FFI_SIGNATURE = -17890115  # 0xFEEF04BD
VS_FFI_STRUCVERSION = 0x00010000
VS_FFI_FILEFLAGSMASK = 0x0000003F
VOS_NT_WINDOWS32 = 0x00040004

null_byte = "\0".encode("ascii")  # str in py2k, bytes in py3k


#
# Set VS_FF_PRERELEASE and DEBUG if Debug
#
def file_flags(debug):
    if debug:
        return 3  # VS_FF_DEBUG | VS_FF_PRERELEASE
    return 0


def file_type(is_dll):
    if is_dll:
        return 2  # VFT_DLL
    return 1  # VFT_APP


def VS_FIXEDFILEINFO(maj, min, sub, build, debug=0, is_dll=1):
    return struct.pack(
        "lllllllllllll",
        VS_FFI_SIGNATURE,  # dwSignature
        VS_FFI_STRUCVERSION,  # dwStrucVersion
        (maj << 16) | min,  # dwFileVersionMS
        (sub << 16) | build,  # dwFileVersionLS
        (maj << 16) | min,  # dwProductVersionMS
        (sub << 16) | build,  # dwProductVersionLS
        VS_FFI_FILEFLAGSMASK,  # dwFileFlagsMask
        file_flags(debug),  # dwFileFlags
        VOS_NT_WINDOWS32,  # dwFileOS
        file_type(is_dll),  # dwFileType
        0x00000000,  # dwFileSubtype
        0x00000000,  # dwFileDateMS
        0x00000000,  # dwFileDateLS
    )


def nullterm(s):
    # get raw bytes for a NULL terminated unicode string.
    if sys.version_info[:2] < (3, 7):
        return (str(s) + "\0").encode("unicode-internal")
    else:
        return (str(s) + "\0").encode("utf-16le")


def pad32(s, extra=2):
    # extra is normally 2 to deal with wLength
    l = 4 - ((len(s) + extra) & 3)
    if l < 4:
        return s + (null_byte * l)
    return s


def addlen(s):
    return struct.pack("h", len(s) + 2) + s


def String(key, value):
    key = nullterm(key)
    value = nullterm(value)
    result = struct.pack("hh", len(value) // 2, 1)  # wValueLength, wType
    result = result + key
    result = pad32(result) + value
    return addlen(result)


def StringTable(key, data):
    key = nullterm(key)
    result = struct.pack("hh", 0, 1)  # wValueLength, wType
    result = result + key
    for k, v in data.items():
        result = result + String(k, v)
        result = pad32(result)
    return addlen(result)


def StringFileInfo(data):
    result = struct.pack("hh", 0, 1)  # wValueLength, wType
    result = result + nullterm("StringFileInfo")
    #  result = pad32(result) + StringTable('040904b0', data)
    result = pad32(result) + StringTable("040904E4", data)
    return addlen(result)


def Var(key, value):
    result = struct.pack("hh", len(value), 0)  # wValueLength, wType
    result = result + nullterm(key)
    result = pad32(result) + value
    return addlen(result)


def VarFileInfo(data):
    result = struct.pack("hh", 0, 1)  # wValueLength, wType
    result = result + nullterm("VarFileInfo")
    result = pad32(result)
    for k, v in data.items():
        result = result + Var(k, v)
    return addlen(result)


def VS_VERSION_INFO(maj, min, sub, build, sdata, vdata, debug=0, is_dll=1):
    ffi = VS_FIXEDFILEINFO(maj, min, sub, build, debug, is_dll)
    result = struct.pack("hh", len(ffi), 0)  # wValueLength, wType
    result = result + nullterm("VS_VERSION_INFO")
    result = pad32(result) + ffi
    result = pad32(result) + StringFileInfo(sdata) + VarFileInfo(vdata)
    return addlen(result)


def stamp(pathname, options):
    # For some reason, the API functions report success if the file is open
    # but doesnt work!  Try and open the file for writing, just to see if it is
    # likely the stamp will work!
    try:
        f = open(pathname, "a+b")
        f.close()
    except IOError as why:
        print("WARNING: File %s could not be opened - %s" % (pathname, why))

    ver = options.version
    try:
        bits = [int(i) for i in ver.split(".")]
        vmaj, vmin, vsub, vbuild = bits
    except (IndexError, TypeError, ValueError):
        raise ValueError("--version must be a.b.c.d (all integers) - got %r" % ver)

    ifn = options.internal_name
    if not ifn:
        ifn = os.path.basename(pathname)
    ofn = options.original_filename
    if ofn is None:
        ofn = os.path.basename(pathname)

    sdata = {
        "Comments": options.comments,
        "CompanyName": options.company,
        "FileDescription": options.description,
        "FileVersion": ver,
        "InternalName": ifn,
        "LegalCopyright": options.copyright,
        "LegalTrademarks": options.trademarks,
        "OriginalFilename": ofn,
        "ProductName": options.product,
        "ProductVersion": ver,
    }
    vdata = {
        "Translation": struct.pack("hh", 0x409, 1252),
    }
    is_dll = options.dll
    if is_dll is None:
        is_dll = os.path.splitext(pathname)[1].lower() in ".dll .pyd".split()
    is_debug = options.debug
    if is_debug is None:
        is_debug = os.path.splitext(pathname)[0].lower().endswith("_d")
    # convert None to blank strings
    for k, v in list(sdata.items()):
        if v is None:
            sdata[k] = ""
    vs = VS_VERSION_INFO(vmaj, vmin, vsub, vbuild, sdata, vdata, is_debug, is_dll)

    h = BeginUpdateResource(pathname, 0)
    UpdateResource(h, 16, 1, vs)
    EndUpdateResource(h, 0)

    if options.verbose:
        print("Stamped:", pathname)


if __name__ == "__main__":
    parser = optparse.OptionParser("%prog [options] filespec ...", description=__doc__)

    parser.add_option(
        "-q",
        "--quiet",
        action="store_false",
        dest="verbose",
        default=True,
        help="don't print status messages to stdout",
    )
    parser.add_option(
        "", "--version", default="0.0.0.0", help="The version number as m.n.s.b"
    )
    parser.add_option(
        "",
        "--dll",
        help="""Stamp the file as a DLL.  Default is to look at the
                            file extension for .dll or .pyd.""",
    )
    parser.add_option("", "--debug", help="""Stamp the file as a debug binary.""")
    parser.add_option("", "--product", help="""The product name to embed.""")
    parser.add_option("", "--company", help="""The company name to embed.""")
    parser.add_option("", "--trademarks", help="The trademark string to embed.")
    parser.add_option("", "--comments", help="The comments string to embed.")
    parser.add_option(
        "", "--copyright", help="""The copyright message string to embed."""
    )
    parser.add_option(
        "", "--description", metavar="DESC", help="The description to embed."
    )
    parser.add_option(
        "",
        "--internal-name",
        metavar="NAME",
        help="""The internal filename to embed. If not specified
                         the base filename is used.""",
    )
    parser.add_option(
        "",
        "--original-filename",
        help="""The original filename to embed. If not specified
                            the base filename is used.""",
    )

    options, args = parser.parse_args()
    if not args:
        parser.error("You must supply a file to stamp.  Use --help for details.")

    for g in args:
        for f in glob.glob(g):
            stamp(f, options)
