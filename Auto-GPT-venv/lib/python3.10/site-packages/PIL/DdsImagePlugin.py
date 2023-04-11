"""
A Pillow loader for .dds files (S3TC-compressed aka DXTC)
Jerome Leclanche <jerome@leclan.ch>

Documentation:
  https://web.archive.org/web/20170802060935/http://oss.sgi.com/projects/ogl-sample/registry/EXT/texture_compression_s3tc.txt

The contents of this file are hereby released in the public domain (CC0)
Full text of the CC0 license:
  https://creativecommons.org/publicdomain/zero/1.0/
"""

import struct
from io import BytesIO

from . import Image, ImageFile
from ._binary import o32le as o32

# Magic ("DDS ")
DDS_MAGIC = 0x20534444

# DDS flags
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSD_DEPTH = 0x800000

# DDS caps
DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000

DDSCAPS2_CUBEMAP = 0x200
DDSCAPS2_CUBEMAP_POSITIVEX = 0x400
DDSCAPS2_CUBEMAP_NEGATIVEX = 0x800
DDSCAPS2_CUBEMAP_POSITIVEY = 0x1000
DDSCAPS2_CUBEMAP_NEGATIVEY = 0x2000
DDSCAPS2_CUBEMAP_POSITIVEZ = 0x4000
DDSCAPS2_CUBEMAP_NEGATIVEZ = 0x8000
DDSCAPS2_VOLUME = 0x200000

# Pixel Format
DDPF_ALPHAPIXELS = 0x1
DDPF_ALPHA = 0x2
DDPF_FOURCC = 0x4
DDPF_PALETTEINDEXED8 = 0x20
DDPF_RGB = 0x40
DDPF_LUMINANCE = 0x20000


# dds.h

DDS_FOURCC = DDPF_FOURCC
DDS_RGB = DDPF_RGB
DDS_RGBA = DDPF_RGB | DDPF_ALPHAPIXELS
DDS_LUMINANCE = DDPF_LUMINANCE
DDS_LUMINANCEA = DDPF_LUMINANCE | DDPF_ALPHAPIXELS
DDS_ALPHA = DDPF_ALPHA
DDS_PAL8 = DDPF_PALETTEINDEXED8

DDS_HEADER_FLAGS_TEXTURE = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
DDS_HEADER_FLAGS_MIPMAP = DDSD_MIPMAPCOUNT
DDS_HEADER_FLAGS_VOLUME = DDSD_DEPTH
DDS_HEADER_FLAGS_PITCH = DDSD_PITCH
DDS_HEADER_FLAGS_LINEARSIZE = DDSD_LINEARSIZE

DDS_HEIGHT = DDSD_HEIGHT
DDS_WIDTH = DDSD_WIDTH

DDS_SURFACE_FLAGS_TEXTURE = DDSCAPS_TEXTURE
DDS_SURFACE_FLAGS_MIPMAP = DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
DDS_SURFACE_FLAGS_CUBEMAP = DDSCAPS_COMPLEX

DDS_CUBEMAP_POSITIVEX = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX
DDS_CUBEMAP_NEGATIVEX = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEX
DDS_CUBEMAP_POSITIVEY = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEY
DDS_CUBEMAP_NEGATIVEY = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEY
DDS_CUBEMAP_POSITIVEZ = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEZ
DDS_CUBEMAP_NEGATIVEZ = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEZ


# DXT1
DXT1_FOURCC = 0x31545844

# DXT3
DXT3_FOURCC = 0x33545844

# DXT5
DXT5_FOURCC = 0x35545844


# dxgiformat.h

DXGI_FORMAT_R8G8B8A8_TYPELESS = 27
DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29
DXGI_FORMAT_BC5_TYPELESS = 82
DXGI_FORMAT_BC5_UNORM = 83
DXGI_FORMAT_BC5_SNORM = 84
DXGI_FORMAT_BC6H_UF16 = 95
DXGI_FORMAT_BC6H_SF16 = 96
DXGI_FORMAT_BC7_TYPELESS = 97
DXGI_FORMAT_BC7_UNORM = 98
DXGI_FORMAT_BC7_UNORM_SRGB = 99


class DdsImageFile(ImageFile.ImageFile):
    format = "DDS"
    format_description = "DirectDraw Surface"

    def _open(self):
        if not _accept(self.fp.read(4)):
            msg = "not a DDS file"
            raise SyntaxError(msg)
        (header_size,) = struct.unpack("<I", self.fp.read(4))
        if header_size != 124:
            msg = f"Unsupported header size {repr(header_size)}"
            raise OSError(msg)
        header_bytes = self.fp.read(header_size - 4)
        if len(header_bytes) != 120:
            msg = f"Incomplete header: {len(header_bytes)} bytes"
            raise OSError(msg)
        header = BytesIO(header_bytes)

        flags, height, width = struct.unpack("<3I", header.read(12))
        self._size = (width, height)
        self.mode = "RGBA"

        pitch, depth, mipmaps = struct.unpack("<3I", header.read(12))
        struct.unpack("<11I", header.read(44))  # reserved

        # pixel format
        pfsize, pfflags = struct.unpack("<2I", header.read(8))
        fourcc = header.read(4)
        (bitcount,) = struct.unpack("<I", header.read(4))
        masks = struct.unpack("<4I", header.read(16))
        if pfflags & DDPF_LUMINANCE:
            # Texture contains uncompressed L or LA data
            if pfflags & DDPF_ALPHAPIXELS:
                self.mode = "LA"
            else:
                self.mode = "L"

            self.tile = [("raw", (0, 0) + self.size, 0, (self.mode, 0, 1))]
        elif pfflags & DDPF_RGB:
            # Texture contains uncompressed RGB data
            masks = {mask: ["R", "G", "B", "A"][i] for i, mask in enumerate(masks)}
            rawmode = ""
            if pfflags & DDPF_ALPHAPIXELS:
                rawmode += masks[0xFF000000]
            else:
                self.mode = "RGB"
            rawmode += masks[0xFF0000] + masks[0xFF00] + masks[0xFF]

            self.tile = [("raw", (0, 0) + self.size, 0, (rawmode[::-1], 0, 1))]
        else:
            data_start = header_size + 4
            n = 0
            if fourcc == b"DXT1":
                self.pixel_format = "DXT1"
                n = 1
            elif fourcc == b"DXT3":
                self.pixel_format = "DXT3"
                n = 2
            elif fourcc == b"DXT5":
                self.pixel_format = "DXT5"
                n = 3
            elif fourcc == b"ATI1":
                self.pixel_format = "BC4"
                n = 4
                self.mode = "L"
            elif fourcc == b"ATI2":
                self.pixel_format = "BC5"
                n = 5
                self.mode = "RGB"
            elif fourcc == b"BC5S":
                self.pixel_format = "BC5S"
                n = 5
                self.mode = "RGB"
            elif fourcc == b"DX10":
                data_start += 20
                # ignoring flags which pertain to volume textures and cubemaps
                (dxgi_format,) = struct.unpack("<I", self.fp.read(4))
                self.fp.read(16)
                if dxgi_format in (DXGI_FORMAT_BC5_TYPELESS, DXGI_FORMAT_BC5_UNORM):
                    self.pixel_format = "BC5"
                    n = 5
                    self.mode = "RGB"
                elif dxgi_format == DXGI_FORMAT_BC5_SNORM:
                    self.pixel_format = "BC5S"
                    n = 5
                    self.mode = "RGB"
                elif dxgi_format == DXGI_FORMAT_BC6H_UF16:
                    self.pixel_format = "BC6H"
                    n = 6
                    self.mode = "RGB"
                elif dxgi_format == DXGI_FORMAT_BC6H_SF16:
                    self.pixel_format = "BC6HS"
                    n = 6
                    self.mode = "RGB"
                elif dxgi_format in (DXGI_FORMAT_BC7_TYPELESS, DXGI_FORMAT_BC7_UNORM):
                    self.pixel_format = "BC7"
                    n = 7
                elif dxgi_format == DXGI_FORMAT_BC7_UNORM_SRGB:
                    self.pixel_format = "BC7"
                    self.info["gamma"] = 1 / 2.2
                    n = 7
                elif dxgi_format in (
                    DXGI_FORMAT_R8G8B8A8_TYPELESS,
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                ):
                    self.tile = [("raw", (0, 0) + self.size, 0, ("RGBA", 0, 1))]
                    if dxgi_format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
                        self.info["gamma"] = 1 / 2.2
                    return
                else:
                    msg = f"Unimplemented DXGI format {dxgi_format}"
                    raise NotImplementedError(msg)
            else:
                msg = f"Unimplemented pixel format {repr(fourcc)}"
                raise NotImplementedError(msg)

            self.tile = [
                ("bcn", (0, 0) + self.size, data_start, (n, self.pixel_format))
            ]

    def load_seek(self, pos):
        pass


def _save(im, fp, filename):
    if im.mode not in ("RGB", "RGBA", "L", "LA"):
        msg = f"cannot write mode {im.mode} as DDS"
        raise OSError(msg)

    rawmode = im.mode
    masks = [0xFF0000, 0xFF00, 0xFF]
    if im.mode in ("L", "LA"):
        pixel_flags = DDPF_LUMINANCE
    else:
        pixel_flags = DDPF_RGB
        rawmode = rawmode[::-1]
    if im.mode in ("LA", "RGBA"):
        pixel_flags |= DDPF_ALPHAPIXELS
        masks.append(0xFF000000)

    bitcount = len(masks) * 8
    while len(masks) < 4:
        masks.append(0)

    fp.write(
        o32(DDS_MAGIC)
        + o32(124)  # header size
        + o32(
            DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT
        )  # flags
        + o32(im.height)
        + o32(im.width)
        + o32((im.width * bitcount + 7) // 8)  # pitch
        + o32(0)  # depth
        + o32(0)  # mipmaps
        + o32(0) * 11  # reserved
        + o32(32)  # pfsize
        + o32(pixel_flags)  # pfflags
        + o32(0)  # fourcc
        + o32(bitcount)  # bitcount
        + b"".join(o32(mask) for mask in masks)  # rgbabitmask
        + o32(DDSCAPS_TEXTURE)  # dwCaps
        + o32(0)  # dwCaps2
        + o32(0)  # dwCaps3
        + o32(0)  # dwCaps4
        + o32(0)  # dwReserved2
    )
    if im.mode == "RGBA":
        r, g, b, a = im.split()
        im = Image.merge("RGBA", (a, r, g, b))
    ImageFile._save(im, fp, [("raw", (0, 0) + im.size, 0, (rawmode, 0, 1))])


def _accept(prefix):
    return prefix[:4] == b"DDS "


Image.register_open(DdsImageFile.format, DdsImageFile, _accept)
Image.register_save(DdsImageFile.format, _save)
Image.register_extension(DdsImageFile.format, ".dds")
