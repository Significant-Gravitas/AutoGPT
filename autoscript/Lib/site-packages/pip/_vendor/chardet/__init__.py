######################## BEGIN LICENSE BLOCK ########################
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
# 02110-1301  USA
######################### END LICENSE BLOCK #########################

from .enums import InputState
from .universaldetector import UniversalDetector
from .version import VERSION, __version__

__all__ = ["UniversalDetector", "detect", "detect_all", "__version__", "VERSION"]


def detect(byte_str):
    """
    Detect the encoding of the given byte string.

    :param byte_str:     The byte sequence to examine.
    :type byte_str:      ``bytes`` or ``bytearray``
    """
    if not isinstance(byte_str, bytearray):
        if not isinstance(byte_str, bytes):
            raise TypeError(
                f"Expected object of type bytes or bytearray, got: {type(byte_str)}"
            )
        byte_str = bytearray(byte_str)
    detector = UniversalDetector()
    detector.feed(byte_str)
    return detector.close()


def detect_all(byte_str, ignore_threshold=False):
    """
    Detect all the possible encodings of the given byte string.

    :param byte_str:          The byte sequence to examine.
    :type byte_str:           ``bytes`` or ``bytearray``
    :param ignore_threshold:  Include encodings that are below
                              ``UniversalDetector.MINIMUM_THRESHOLD``
                              in results.
    :type ignore_threshold:   ``bool``
    """
    if not isinstance(byte_str, bytearray):
        if not isinstance(byte_str, bytes):
            raise TypeError(
                f"Expected object of type bytes or bytearray, got: {type(byte_str)}"
            )
        byte_str = bytearray(byte_str)

    detector = UniversalDetector()
    detector.feed(byte_str)
    detector.close()

    if detector.input_state == InputState.HIGH_BYTE:
        results = []
        probers = []
        for prober in detector.charset_probers:
            if hasattr(prober, "probers"):
                probers.extend(p for p in prober.probers)
            else:
                probers.append(prober)
        for prober in probers:
            if ignore_threshold or prober.get_confidence() > detector.MINIMUM_THRESHOLD:
                charset_name = prober.charset_name or ""
                lower_charset_name = charset_name.lower()
                # Use Windows encoding name instead of ISO-8859 if we saw any
                # extra Windows-specific bytes
                if lower_charset_name.startswith("iso-8859") and detector.has_win_bytes:
                    charset_name = detector.ISO_WIN_MAP.get(
                        lower_charset_name, charset_name
                    )
                results.append(
                    {
                        "encoding": charset_name,
                        "confidence": prober.get_confidence(),
                        "language": prober.language,
                    }
                )
        if len(results) > 0:
            return sorted(results, key=lambda result: -result["confidence"])

    return [detector.result]
