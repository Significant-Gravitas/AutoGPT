# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

"""Common zone-related types."""

# This is a separate file to avoid import circularity between dns.zone and
# the implementation of the ZONEMD type.

import hashlib

import dns.enum


class DigestScheme(dns.enum.IntEnum):
    """ZONEMD Scheme"""

    SIMPLE = 1

    @classmethod
    def _maximum(cls):
        return 255


class DigestHashAlgorithm(dns.enum.IntEnum):
    """ZONEMD Hash Algorithm"""

    SHA384 = 1
    SHA512 = 2

    @classmethod
    def _maximum(cls):
        return 255


_digest_hashers = {
    DigestHashAlgorithm.SHA384: hashlib.sha384,
    DigestHashAlgorithm.SHA512: hashlib.sha512,
}
