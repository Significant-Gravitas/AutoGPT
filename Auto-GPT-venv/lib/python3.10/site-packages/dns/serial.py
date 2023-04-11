# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

"""Serial Number Arthimetic from RFC 1982"""


class Serial:
    def __init__(self, value: int, bits: int = 32):
        self.value = value % 2**bits
        self.bits = bits

    def __repr__(self):
        return f"dns.serial.Serial({self.value}, {self.bits})"

    def __eq__(self, other):
        if isinstance(other, int):
            other = Serial(other, self.bits)
        elif not isinstance(other, Serial) or other.bits != self.bits:
            return NotImplemented
        return self.value == other.value

    def __ne__(self, other):
        if isinstance(other, int):
            other = Serial(other, self.bits)
        elif not isinstance(other, Serial) or other.bits != self.bits:
            return NotImplemented
        return self.value != other.value

    def __lt__(self, other):
        if isinstance(other, int):
            other = Serial(other, self.bits)
        elif not isinstance(other, Serial) or other.bits != self.bits:
            return NotImplemented
        if self.value < other.value and other.value - self.value < 2 ** (self.bits - 1):
            return True
        elif self.value > other.value and self.value - other.value > 2 ** (
            self.bits - 1
        ):
            return True
        else:
            return False

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if isinstance(other, int):
            other = Serial(other, self.bits)
        elif not isinstance(other, Serial) or other.bits != self.bits:
            return NotImplemented
        if self.value < other.value and other.value - self.value > 2 ** (self.bits - 1):
            return True
        elif self.value > other.value and self.value - other.value < 2 ** (
            self.bits - 1
        ):
            return True
        else:
            return False

    def __ge__(self, other):
        return self == other or self > other

    def __add__(self, other):
        v = self.value
        if isinstance(other, Serial):
            delta = other.value
        elif isinstance(other, int):
            delta = other
        else:
            raise ValueError
        if abs(delta) > (2 ** (self.bits - 1) - 1):
            raise ValueError
        v += delta
        v = v % 2**self.bits
        return Serial(v, self.bits)

    def __iadd__(self, other):
        v = self.value
        if isinstance(other, Serial):
            delta = other.value
        elif isinstance(other, int):
            delta = other
        else:
            raise ValueError
        if abs(delta) > (2 ** (self.bits - 1) - 1):
            raise ValueError
        v += delta
        v = v % 2**self.bits
        self.value = v
        return self

    def __sub__(self, other):
        v = self.value
        if isinstance(other, Serial):
            delta = other.value
        elif isinstance(other, int):
            delta = other
        else:
            raise ValueError
        if abs(delta) > (2 ** (self.bits - 1) - 1):
            raise ValueError
        v -= delta
        v = v % 2**self.bits
        return Serial(v, self.bits)

    def __isub__(self, other):
        v = self.value
        if isinstance(other, Serial):
            delta = other.value
        elif isinstance(other, int):
            delta = other
        else:
            raise ValueError
        if abs(delta) > (2 ** (self.bits - 1) - 1):
            raise ValueError
        v -= delta
        v = v % 2**self.bits
        self.value = v
        return self
