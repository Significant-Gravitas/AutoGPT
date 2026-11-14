"""The shared libraries a Linux executable asks the loader for (its DT_NEEDED
entries), read from the file itself: what `readelf -d` prints, without needing
binutils on the build machine or a Linux to run it on.

Only what the bundle's own executables are: 64-bit, little-endian.
"""

from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"\x7fELF"
ELF64, LITTLE_ENDIAN = 2, 1
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
DYNAMIC_ENTRY = struct.Struct("<qQ")
SHT_DYNAMIC = 6
DT_NULL, DT_NEEDED = 0, 1


class NotElf(ValueError):
    pass


def needed(path: Path) -> list[str]:
    return needed_in(path.read_bytes(), str(path))


def needed_in(image: bytes, name: str = "the file") -> list[str]:
    if image[:4] != MAGIC or len(image) < 64:
        raise NotElf(f"{name} is not an ELF executable")
    if image[4] != ELF64 or image[5] != LITTLE_ENDIAN:
        raise NotElf(f"{name} is not a 64-bit little-endian ELF executable")
    (section_offset,) = struct.unpack_from("<Q", image, 0x28)
    entry_size, count = struct.unpack_from("<HH", image, 0x3A)
    sections = [
        SECTION_HEADER.unpack_from(image, section_offset + index * entry_size)
        for index in range(count)
    ]
    libraries = []
    for _, kind, _, _, offset, size, link, *_ in sections:
        if kind != SHT_DYNAMIC:
            continue
        strings = sections[link][4]
        for position in range(offset, offset + size, DYNAMIC_ENTRY.size):
            tag, value = DYNAMIC_ENTRY.unpack_from(image, position)
            if tag == DT_NULL:
                break
            if tag == DT_NEEDED:
                start = strings + value
                libraries.append(image[start : image.index(b"\0", start)].decode())
    return libraries
