from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import uuid

import requests


def read_file(blobpath: str) -> bytes:
    if not blobpath.startswith("http://") and not blobpath.startswith("https://"):
        try:
            import blobfile
        except ImportError:
            raise ImportError(
                "blobfile is not installed. Please install it by running `pip install blobfile`."
            )
        with blobfile.BlobFile(blobpath, "rb") as f:
            return f.read()
    # avoiding blobfile for public files helps avoid auth issues, like MFA prompts
    return requests.get(blobpath).content


def read_file_cached(blobpath: str) -> bytes:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")

    if cache_dir == "":
        # disable caching
        return read_file(blobpath)

    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    contents = read_file(blobpath)

    os.makedirs(cache_dir, exist_ok=True)
    tmp_filename = cache_path + "." + str(uuid.uuid4()) + ".tmp"
    with open(tmp_filename, "wb") as f:
        f.write(contents)
    os.rename(tmp_filename, cache_path)

    return contents


def data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file: str, encoder_json_file: str
) -> dict[bytes, int]:
    # NB: do not add caching to this function
    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]

    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
    n = 0
    for b in range(2**8):
        if b not in rank_to_intbyte:
            rank_to_intbyte.append(b)
            data_gym_byte_to_byte[chr(2**8 + n)] = b
            n += 1
    assert len(rank_to_intbyte) == 2**8

    # vocab_bpe contains the merges along with associated ranks
    vocab_bpe_contents = read_file_cached(vocab_bpe_file).decode()
    bpe_merges = [tuple(merge_str.split()) for merge_str in vocab_bpe_contents.split("\n")[1:-1]]

    def decode_data_gym(value: str) -> bytes:
        return bytes(data_gym_byte_to_byte[b] for b in value)

    # add the single byte tokens
    bpe_ranks = {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
    # add the merged tokens
    n = len(bpe_ranks)
    for first, second in bpe_merges:
        bpe_ranks[decode_data_gym(first) + decode_data_gym(second)] = n
        n += 1

    # check that the encoder file matches the merges file
    # this sanity check is important since tiktoken assumes that ranks are ordered the same
    # as merge priority
    encoder_json = json.loads(read_file_cached(encoder_json_file))
    encoder_json_loaded = {decode_data_gym(k): v for k, v in encoder_json.items()}
    # drop these two special tokens if present, since they're not mergeable bpe tokens
    encoder_json_loaded.pop(b"<|endoftext|>", None)
    encoder_json_loaded.pop(b"<|startoftext|>", None)
    assert bpe_ranks == encoder_json_loaded

    return bpe_ranks


def dump_tiktoken_bpe(bpe_ranks: dict[bytes, int], tiktoken_bpe_file: str) -> None:
    try:
        import blobfile
    except ImportError:
        raise ImportError(
            "blobfile is not installed. Please install it by running `pip install blobfile`."
        )
    with blobfile.BlobFile(tiktoken_bpe_file, "wb") as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            f.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")


def load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
    # NB: do not add caching to this function
    contents = read_file_cached(tiktoken_bpe_file)
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }
