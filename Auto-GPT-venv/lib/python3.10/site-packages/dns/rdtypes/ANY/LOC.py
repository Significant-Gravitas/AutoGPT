# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2007, 2009-2011 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import struct

import dns.exception
import dns.immutable
import dns.rdata


_pows = tuple(10**i for i in range(0, 11))

# default values are in centimeters
_default_size = 100.0
_default_hprec = 1000000.0
_default_vprec = 1000.0

# for use by from_wire()
_MAX_LATITUDE = 0x80000000 + 90 * 3600000
_MIN_LATITUDE = 0x80000000 - 90 * 3600000
_MAX_LONGITUDE = 0x80000000 + 180 * 3600000
_MIN_LONGITUDE = 0x80000000 - 180 * 3600000


def _exponent_of(what, desc):
    if what == 0:
        return 0
    exp = None
    for (i, pow) in enumerate(_pows):
        if what < pow:
            exp = i - 1
            break
    if exp is None or exp < 0:
        raise dns.exception.SyntaxError("%s value out of bounds" % desc)
    return exp


def _float_to_tuple(what):
    if what < 0:
        sign = -1
        what *= -1
    else:
        sign = 1
    what = round(what * 3600000)
    degrees = int(what // 3600000)
    what -= degrees * 3600000
    minutes = int(what // 60000)
    what -= minutes * 60000
    seconds = int(what // 1000)
    what -= int(seconds * 1000)
    what = int(what)
    return (degrees, minutes, seconds, what, sign)


def _tuple_to_float(what):
    value = float(what[0])
    value += float(what[1]) / 60.0
    value += float(what[2]) / 3600.0
    value += float(what[3]) / 3600000.0
    return float(what[4]) * value


def _encode_size(what, desc):
    what = int(what)
    exponent = _exponent_of(what, desc) & 0xF
    base = what // pow(10, exponent) & 0xF
    return base * 16 + exponent


def _decode_size(what, desc):
    exponent = what & 0x0F
    if exponent > 9:
        raise dns.exception.FormError("bad %s exponent" % desc)
    base = (what & 0xF0) >> 4
    if base > 9:
        raise dns.exception.FormError("bad %s base" % desc)
    return base * pow(10, exponent)


def _check_coordinate_list(value, low, high):
    if value[0] < low or value[0] > high:
        raise ValueError(f"not in range [{low}, {high}]")
    if value[1] < 0 or value[1] > 59:
        raise ValueError("bad minutes value")
    if value[2] < 0 or value[2] > 59:
        raise ValueError("bad seconds value")
    if value[3] < 0 or value[3] > 999:
        raise ValueError("bad milliseconds value")
    if value[4] != 1 and value[4] != -1:
        raise ValueError("bad hemisphere value")


@dns.immutable.immutable
class LOC(dns.rdata.Rdata):

    """LOC record"""

    # see: RFC 1876

    __slots__ = [
        "latitude",
        "longitude",
        "altitude",
        "size",
        "horizontal_precision",
        "vertical_precision",
    ]

    def __init__(
        self,
        rdclass,
        rdtype,
        latitude,
        longitude,
        altitude,
        size=_default_size,
        hprec=_default_hprec,
        vprec=_default_vprec,
    ):
        """Initialize a LOC record instance.

        The parameters I{latitude} and I{longitude} may be either a 4-tuple
        of integers specifying (degrees, minutes, seconds, milliseconds),
        or they may be floating point values specifying the number of
        degrees. The other parameters are floats. Size, horizontal precision,
        and vertical precision are specified in centimeters."""

        super().__init__(rdclass, rdtype)
        if isinstance(latitude, int):
            latitude = float(latitude)
        if isinstance(latitude, float):
            latitude = _float_to_tuple(latitude)
        _check_coordinate_list(latitude, -90, 90)
        self.latitude = tuple(latitude)
        if isinstance(longitude, int):
            longitude = float(longitude)
        if isinstance(longitude, float):
            longitude = _float_to_tuple(longitude)
        _check_coordinate_list(longitude, -180, 180)
        self.longitude = tuple(longitude)
        self.altitude = float(altitude)
        self.size = float(size)
        self.horizontal_precision = float(hprec)
        self.vertical_precision = float(vprec)

    def to_text(self, origin=None, relativize=True, **kw):
        if self.latitude[4] > 0:
            lat_hemisphere = "N"
        else:
            lat_hemisphere = "S"
        if self.longitude[4] > 0:
            long_hemisphere = "E"
        else:
            long_hemisphere = "W"
        text = "%d %d %d.%03d %s %d %d %d.%03d %s %0.2fm" % (
            self.latitude[0],
            self.latitude[1],
            self.latitude[2],
            self.latitude[3],
            lat_hemisphere,
            self.longitude[0],
            self.longitude[1],
            self.longitude[2],
            self.longitude[3],
            long_hemisphere,
            self.altitude / 100.0,
        )

        # do not print default values
        if (
            self.size != _default_size
            or self.horizontal_precision != _default_hprec
            or self.vertical_precision != _default_vprec
        ):
            text += " {:0.2f}m {:0.2f}m {:0.2f}m".format(
                self.size / 100.0,
                self.horizontal_precision / 100.0,
                self.vertical_precision / 100.0,
            )
        return text

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        latitude = [0, 0, 0, 0, 1]
        longitude = [0, 0, 0, 0, 1]
        size = _default_size
        hprec = _default_hprec
        vprec = _default_vprec

        latitude[0] = tok.get_int()
        t = tok.get_string()
        if t.isdigit():
            latitude[1] = int(t)
            t = tok.get_string()
            if "." in t:
                (seconds, milliseconds) = t.split(".")
                if not seconds.isdigit():
                    raise dns.exception.SyntaxError("bad latitude seconds value")
                latitude[2] = int(seconds)
                l = len(milliseconds)
                if l == 0 or l > 3 or not milliseconds.isdigit():
                    raise dns.exception.SyntaxError("bad latitude milliseconds value")
                if l == 1:
                    m = 100
                elif l == 2:
                    m = 10
                else:
                    m = 1
                latitude[3] = m * int(milliseconds)
                t = tok.get_string()
            elif t.isdigit():
                latitude[2] = int(t)
                t = tok.get_string()
        if t == "S":
            latitude[4] = -1
        elif t != "N":
            raise dns.exception.SyntaxError("bad latitude hemisphere value")

        longitude[0] = tok.get_int()
        t = tok.get_string()
        if t.isdigit():
            longitude[1] = int(t)
            t = tok.get_string()
            if "." in t:
                (seconds, milliseconds) = t.split(".")
                if not seconds.isdigit():
                    raise dns.exception.SyntaxError("bad longitude seconds value")
                longitude[2] = int(seconds)
                l = len(milliseconds)
                if l == 0 or l > 3 or not milliseconds.isdigit():
                    raise dns.exception.SyntaxError("bad longitude milliseconds value")
                if l == 1:
                    m = 100
                elif l == 2:
                    m = 10
                else:
                    m = 1
                longitude[3] = m * int(milliseconds)
                t = tok.get_string()
            elif t.isdigit():
                longitude[2] = int(t)
                t = tok.get_string()
        if t == "W":
            longitude[4] = -1
        elif t != "E":
            raise dns.exception.SyntaxError("bad longitude hemisphere value")

        t = tok.get_string()
        if t[-1] == "m":
            t = t[0:-1]
        altitude = float(t) * 100.0  # m -> cm

        tokens = tok.get_remaining(max_tokens=3)
        if len(tokens) >= 1:
            value = tokens[0].unescape().value
            if value[-1] == "m":
                value = value[0:-1]
            size = float(value) * 100.0  # m -> cm
            if len(tokens) >= 2:
                value = tokens[1].unescape().value
                if value[-1] == "m":
                    value = value[0:-1]
                hprec = float(value) * 100.0  # m -> cm
                if len(tokens) >= 3:
                    value = tokens[2].unescape().value
                    if value[-1] == "m":
                        value = value[0:-1]
                    vprec = float(value) * 100.0  # m -> cm

        # Try encoding these now so we raise if they are bad
        _encode_size(size, "size")
        _encode_size(hprec, "horizontal precision")
        _encode_size(vprec, "vertical precision")

        return cls(rdclass, rdtype, latitude, longitude, altitude, size, hprec, vprec)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        milliseconds = (
            self.latitude[0] * 3600000
            + self.latitude[1] * 60000
            + self.latitude[2] * 1000
            + self.latitude[3]
        ) * self.latitude[4]
        latitude = 0x80000000 + milliseconds
        milliseconds = (
            self.longitude[0] * 3600000
            + self.longitude[1] * 60000
            + self.longitude[2] * 1000
            + self.longitude[3]
        ) * self.longitude[4]
        longitude = 0x80000000 + milliseconds
        altitude = int(self.altitude) + 10000000
        size = _encode_size(self.size, "size")
        hprec = _encode_size(self.horizontal_precision, "horizontal precision")
        vprec = _encode_size(self.vertical_precision, "vertical precision")
        wire = struct.pack(
            "!BBBBIII", 0, size, hprec, vprec, latitude, longitude, altitude
        )
        file.write(wire)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        (
            version,
            size,
            hprec,
            vprec,
            latitude,
            longitude,
            altitude,
        ) = parser.get_struct("!BBBBIII")
        if version != 0:
            raise dns.exception.FormError("LOC version not zero")
        if latitude < _MIN_LATITUDE or latitude > _MAX_LATITUDE:
            raise dns.exception.FormError("bad latitude")
        if latitude > 0x80000000:
            latitude = (latitude - 0x80000000) / 3600000
        else:
            latitude = -1 * (0x80000000 - latitude) / 3600000
        if longitude < _MIN_LONGITUDE or longitude > _MAX_LONGITUDE:
            raise dns.exception.FormError("bad longitude")
        if longitude > 0x80000000:
            longitude = (longitude - 0x80000000) / 3600000
        else:
            longitude = -1 * (0x80000000 - longitude) / 3600000
        altitude = float(altitude) - 10000000.0
        size = _decode_size(size, "size")
        hprec = _decode_size(hprec, "horizontal precision")
        vprec = _decode_size(vprec, "vertical precision")
        return cls(rdclass, rdtype, latitude, longitude, altitude, size, hprec, vprec)

    @property
    def float_latitude(self):
        "latitude as a floating point value"
        return _tuple_to_float(self.latitude)

    @property
    def float_longitude(self):
        "longitude as a floating point value"
        return _tuple_to_float(self.longitude)
