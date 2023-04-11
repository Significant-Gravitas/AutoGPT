# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

from typing import Iterator, Optional, Tuple

import contextlib
import struct

import dns.exception
import dns.name


class Parser:
    def __init__(self, wire: bytes, current: int = 0):
        self.wire = wire
        self.current = 0
        self.end = len(self.wire)
        if current:
            self.seek(current)
        self.furthest = current

    def remaining(self) -> int:
        return self.end - self.current

    def get_bytes(self, size: int) -> bytes:
        assert size >= 0
        if size > self.remaining():
            raise dns.exception.FormError
        output = self.wire[self.current : self.current + size]
        self.current += size
        self.furthest = max(self.furthest, self.current)
        return output

    def get_counted_bytes(self, length_size: int = 1) -> bytes:
        length = int.from_bytes(self.get_bytes(length_size), "big")
        return self.get_bytes(length)

    def get_remaining(self) -> bytes:
        return self.get_bytes(self.remaining())

    def get_uint8(self) -> int:
        return struct.unpack("!B", self.get_bytes(1))[0]

    def get_uint16(self) -> int:
        return struct.unpack("!H", self.get_bytes(2))[0]

    def get_uint32(self) -> int:
        return struct.unpack("!I", self.get_bytes(4))[0]

    def get_uint48(self) -> int:
        return int.from_bytes(self.get_bytes(6), "big")

    def get_struct(self, format: str) -> Tuple:
        return struct.unpack(format, self.get_bytes(struct.calcsize(format)))

    def get_name(self, origin: Optional["dns.name.Name"] = None) -> "dns.name.Name":
        name = dns.name.from_wire_parser(self)
        if origin:
            name = name.relativize(origin)
        return name

    def seek(self, where: int) -> None:
        # Note that seeking to the end is OK!  (If you try to read
        # after such a seek, you'll get an exception as expected.)
        if where < 0 or where > self.end:
            raise dns.exception.FormError
        self.current = where

    @contextlib.contextmanager
    def restrict_to(self, size: int) -> Iterator:
        assert size >= 0
        if size > self.remaining():
            raise dns.exception.FormError
        saved_end = self.end
        try:
            self.end = self.current + size
            yield
            # We make this check here and not in the finally as we
            # don't want to raise if we're already raising for some
            # other reason.
            if self.current != self.end:
                raise dns.exception.FormError
        finally:
            self.end = saved_end

    @contextlib.contextmanager
    def restore_furthest(self) -> Iterator:
        try:
            yield None
        finally:
            self.current = self.furthest
