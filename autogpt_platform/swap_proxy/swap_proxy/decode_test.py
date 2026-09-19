"""The bound is part of the decode: a small bomb per encoding, and proof that
refusing one does not first produce what it stands for."""

import gzip
import tracemalloc
import zlib

import brotli
import pytest
import zstandard

from swap_proxy.decode import DecodedTooLarge, Undecodable, bounded_decode

LIMIT = 1024 * 1024
TEXT = b'{"echo": "ghp_userAsecretvalue0001"}' * 100


def raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


ENCODERS = {
    "gzip": gzip.compress,
    "deflate": zlib.compress,
    "deflateraw": raw_deflate,
    "br": brotli.compress,
    "zstd": zstandard.ZstdCompressor().compress,
}


@pytest.mark.parametrize("encoding", ENCODERS)
def test_a_body_within_the_limit_decodes_to_what_it_was(encoding):
    assert bounded_decode(ENCODERS[encoding](TEXT), encoding, LIMIT) == TEXT
    assert bounded_decode(ENCODERS[encoding](TEXT), encoding, len(TEXT)) == TEXT


@pytest.mark.parametrize("encoding", ENCODERS)
def test_one_byte_over_the_limit_is_too_large(encoding):
    with pytest.raises(DecodedTooLarge):
        bounded_decode(ENCODERS[encoding](TEXT), encoding, len(TEXT) - 1)


@pytest.mark.parametrize("encoding", ENCODERS)
def test_a_bomb_is_refused_without_being_decoded_first(encoding):
    """A few kilobytes on the wire standing for 64 MiB: refused, and the
    memory it took stays near the limit, nowhere near the decoded size."""
    bomb = ENCODERS[encoding](bytes(64 * 1024 * 1024))
    assert len(bomb) < LIMIT
    tracemalloc.start()
    try:
        with pytest.raises(DecodedTooLarge):
            bounded_decode(bomb, encoding, LIMIT)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * LIMIT


@pytest.mark.parametrize("encoding", ["", "identity", "none", " Identity "])
def test_an_unencoded_body_is_bounded_by_its_own_length(encoding):
    assert bounded_decode(TEXT, encoding, len(TEXT)) == TEXT
    with pytest.raises(DecodedTooLarge):
        bounded_decode(TEXT, encoding, len(TEXT) - 1)


@pytest.mark.parametrize("encoding", ["compress", "gzip, br", "x-made-up"])
def test_an_encoding_that_cannot_be_decoded_within_a_bound_is_undecodable(encoding):
    with pytest.raises(Undecodable):
        bounded_decode(b"\x1f\x8b whatever", encoding, LIMIT)


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "zstd"])
def test_corrupt_data_is_undecodable(encoding):
    with pytest.raises(Undecodable):
        bounded_decode(b"this is not compressed data at all", encoding, LIMIT)


def test_a_truncated_brotli_stream_is_undecodable():
    with pytest.raises(Undecodable):
        bounded_decode(brotli.compress(TEXT)[:-4], "br", LIMIT)


@pytest.mark.parametrize("encoding", ENCODERS)
def test_an_empty_body_is_empty(encoding):
    assert bounded_decode(b"", encoding, LIMIT) == b""
