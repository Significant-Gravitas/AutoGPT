"""The tenant boundary on its own: every way ``authenticate`` says no, and
what a record that says nothing about swapping is taken to mean."""

import hashlib
import json
import logging
from typing import Any

import pytest

from swap_proxy.owners import CREDENTIAL_KEY_PREFIX, Owner, OwnerDirectory

USERNAME = "box-" + "a" * 16
SECRET = "secret-a"


def record(**overrides: Any) -> str:
    fields = {
        "owner": "session:s-a",
        "user_id": "user-a",
        "sandbox_id": "sb-1",
        "swaps": True,
        "secret_sha256": hashlib.sha256(SECRET.encode()).hexdigest(),
    }
    fields.update(overrides)
    return json.dumps({k: v for k, v in fields.items() if v is not ...})


class Reader:
    def __init__(self, raw: Any = None, error: Exception | None = None):
        self.raw, self.error = raw, error
        self.asked: list[str] = []

    async def get(self, name: str):
        self.asked.append(name)
        if self.error:
            raise self.error
        return self.raw


async def authenticate(raw: Any, username: str = USERNAME, secret: str = SECRET):
    return await OwnerDirectory(Reader(raw)).authenticate(username, secret)


async def test_the_right_secret_for_a_minted_username_is_its_owner():
    reader = Reader(record())
    owner = await OwnerDirectory(reader).authenticate(USERNAME, SECRET)
    assert owner == Owner("session:s-a", "user-a", "sb-1", swaps=True, box=USERNAME)
    assert owner is not None and owner.swap_user_id == "user-a"
    assert reader.asked == [CREDENTIAL_KEY_PREFIX + USERNAME]


async def test_a_record_stored_as_bytes_reads_the_same():
    owner = await authenticate(record().encode())
    assert owner is not None and owner.label == "session:s-a"


@pytest.mark.parametrize(
    "username, secret",
    [
        ("box-" + "A" * 16, SECRET),  # not the shape the backend mints
        ("box-" + "a" * 15, SECRET),
        (USERNAME + "\n", SECRET),
        (CREDENTIAL_KEY_PREFIX + USERNAME, SECRET),
        ("", SECRET),
        (USERNAME, ""),
    ],
)
async def test_a_malformed_pair_is_refused_without_touching_redis(username, secret):
    reader = Reader(record())
    assert await OwnerDirectory(reader).authenticate(username, secret) is None
    assert reader.asked == []


async def test_a_redis_that_cannot_be_asked_refuses_the_connection(caplog):
    directory = OwnerDirectory(Reader(error=ConnectionError("redis is down")))
    with caplog.at_level(logging.ERROR, logger="swap_proxy.owners"):
        assert await directory.authenticate(USERNAME, SECRET) is None
    assert "refusing the connection" in caplog.text


@pytest.mark.parametrize("raw", [None, "", b""])
async def test_a_username_nobody_minted_is_refused(raw):
    assert await authenticate(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '"a string"',
        "null",
        record(secret_sha256=...),
        record(owner=...),
    ],
    ids=["not json", "a list", "a string", "null", "no digest", "no owner"],
)
async def test_a_malformed_record_is_refused(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="swap_proxy.owners"):
        assert await authenticate(raw) is None
    assert "Malformed credential record" in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize("secret", ["wrong", SECRET + "x", SECRET.upper()])
async def test_the_wrong_secret_is_refused(secret):
    assert await authenticate(record(), secret=secret) is None


async def test_a_secret_is_not_its_own_digest():
    """Whoever reads the record from Redis holds the digest, not the secret."""
    digest = hashlib.sha256(SECRET.encode()).hexdigest()
    assert await authenticate(record(), secret=digest) is None


async def test_a_record_without_swaps_swaps_nothing():
    """What the base PR's ``_remember`` wrote before the field existed: during
    a rolling deploy the proxy meets such records, and must not swap for them."""
    owner = await authenticate(record(swaps=...))
    assert owner is not None and owner.user_id == "user-a"
    assert owner.swaps is False and owner.swap_user_id is None


@pytest.mark.parametrize("swaps", [False, None, "true", 1, "yes", [True]])
async def test_only_an_explicit_true_swaps(swaps):
    owner = await authenticate(record(swaps=swaps))
    assert owner is not None
    assert owner.swaps is False and owner.swap_user_id is None


async def test_an_owner_without_a_user_has_nobody_to_swap_for():
    owner = await authenticate(record(user_id=None, sandbox_id=""))
    assert owner is not None and owner.swaps is True
    assert owner.user_id is None and owner.sandbox_id is None
    assert owner.swap_user_id is None
