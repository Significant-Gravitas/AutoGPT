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


# ------------------------------------------------------------ whole streams only
#
# zlib stops at the end of the first stream and hands back what it has of a
# cut one without complaint.  A client may read on, so the proxy must too.

TOKEN = b"ghp_userAsecretvalue0001"


def test_every_gzip_member_is_decoded():
    """The second member carries the token: it must be there to be scrubbed."""
    body = gzip.compress(b"first ") + gzip.compress(TOKEN)
    assert bounded_decode(body, "gzip", LIMIT) == b"first " + TOKEN
    assert gzip.decompress(body) == b"first " + TOKEN  # what a client reads


def test_members_together_are_held_to_one_limit():
    body = gzip.compress(TEXT) + gzip.compress(TEXT)
    assert bounded_decode(body, "gzip", 2 * len(TEXT)) == TEXT + TEXT
    with pytest.raises(DecodedTooLarge):
        bounded_decode(body, "gzip", 2 * len(TEXT) - 1)


def test_many_members_each_a_bomb_stay_within_the_limit():
    member = gzip.compress(bytes(LIMIT // 2 + 1))
    with pytest.raises(DecodedTooLarge):
        bounded_decode(member * 3, "gzip", LIMIT)


def test_zero_padding_after_the_last_member_is_allowed():
    assert bounded_decode(gzip.compress(TEXT) + bytes(8), "gzip", LIMIT) == TEXT


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "deflateraw", "br", "zstd"])
def test_data_after_the_end_of_the_stream_is_undecodable(encoding):
    with pytest.raises(Undecodable):
        bounded_decode(ENCODERS[encoding](TEXT) + b"trailing " + TOKEN, encoding, LIMIT)


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "deflateraw", "br", "zstd"])
@pytest.mark.parametrize("cut", [1, 4, 0.5])
def test_a_truncated_stream_is_undecodable(encoding, cut):
    """Cut in the checksum, or halfway through: partial output is not the body."""
    body = ENCODERS[encoding](TEXT + bytes(range(256)) * 64)
    end = int(len(body) * cut) if isinstance(cut, float) else len(body) - cut
    with pytest.raises(Undecodable):
        bounded_decode(body[:end], encoding, LIMIT)


def test_a_truncated_second_member_is_undecodable():
    body = gzip.compress(b"first ") + gzip.compress(TOKEN)[:-4]
    with pytest.raises(Undecodable):
        bounded_decode(body, "gzip", LIMIT)


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_a_body_of_many_tiny_streams_is_undecodable(encoding):
    """Each member or frame is a decoder of its own: a quarter of a million
    empty ones would hold the event loop.  A few are fine."""
    empty = ENCODERS[encoding](b"")
    assert bounded_decode(empty * 64, encoding, LIMIT) == b""
    with pytest.raises(Undecodable):
        bounded_decode(empty * 65, encoding, LIMIT)
    with pytest.raises(Undecodable):
        bounded_decode(empty * (5 * 1024 * 1024 // len(empty)), encoding, LIMIT)


def test_a_zstd_body_of_several_frames_decodes_whole():
    frame = ENCODERS["zstd"]
    assert bounded_decode(frame(b"first ") + frame(TOKEN), "zstd", LIMIT) == (
        b"first " + TOKEN
    )
