"""What checking OAuth credentials costs, and where that work runs."""

import asyncio
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.data.auth import oauth
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    OAuthApplicationInfoWithSecret,
    validate_access_token,
    validate_client_credentials,
)

_CLIENT_SECRET = "agpt_secret_for_the_credential_tests"
_SECRET_HASH, _SECRET_SALT = oauth.keysmith.hash_key(_CLIENT_SECRET)


@pytest.mark.parametrize(
    "token", ["", "garbage", "agpt_rt_refresh-token", "agpt_an-api-key"]
)
async def test_a_value_not_in_the_access_token_format_is_never_looked_up(
    mocker: pytest_mock.MockFixture, token: str
) -> None:
    """Every access token carries the prefix, so nothing else can match one."""
    model = mocker.patch("backend.data.auth.oauth.PrismaOAuthAccessToken")

    with pytest.raises(InvalidTokenError):
        await validate_access_token(token)

    model.prisma.assert_not_called()


@pytest.fixture
def stored_app(mocker: pytest_mock.MockFixture) -> OAuthApplicationInfoWithSecret:
    now = datetime.now(UTC)
    app = OAuthApplicationInfoWithSecret(
        id="app-1",
        name="Credential test app",
        client_id="client-1",
        redirect_uris=["https://example.com/callback"],
        grant_types=["authorization_code", "refresh_token"],
        scopes=[APIKeyPermission.READ_GRAPH],
        owner_id="owner-1",
        is_active=True,
        created_at=now,
        updated_at=now,
        client_secret_hash=_SECRET_HASH,
        client_secret_salt=_SECRET_SALT,
    )
    mocker.patch.object(
        oauth, "get_oauth_application_with_secret", AsyncMock(return_value=app)
    )
    return app


async def test_the_client_secret_the_app_was_issued_is_accepted(
    stored_app: OAuthApplicationInfoWithSecret,
) -> None:
    app = await validate_client_credentials("client-1", _CLIENT_SECRET)

    assert app.id == stored_app.id
    assert "client_secret_hash" not in app.model_dump()


async def test_any_other_client_secret_is_refused(
    stored_app: OAuthApplicationInfoWithSecret,
) -> None:
    with pytest.raises(InvalidClientError, match="Invalid client_secret"):
        await validate_client_credentials("client-1", "agpt_not_the_secret")


async def test_the_event_loop_keeps_serving_while_client_secrets_are_checked(
    mocker: pytest_mock.MockFixture, stored_app: OAuthApplicationInfoWithSecret
) -> None:
    """Each Scrypt check takes tens of milliseconds of CPU. Run on the event
    loop, it holds up every other request the worker is serving, so a burst of
    token, introspection or revocation calls stalls the whole API."""
    check_started = threading.Event()
    other_request_served = threading.Event()
    served_during_check: list[bool] = []

    def slow_check(*args: object) -> bool:
        check_started.set()
        served_during_check.append(other_request_served.wait(timeout=5))
        return False

    mocker.patch.object(oauth.keysmith, "verify_key", side_effect=slow_check)

    async def other_request() -> None:
        while not check_started.is_set():
            await asyncio.sleep(0.01)
        other_request_served.set()

    checks = [
        validate_client_credentials("client-1", "agpt_wrong_secret") for _ in range(2)
    ]
    results = await asyncio.wait_for(
        asyncio.gather(*checks, other_request(), return_exceptions=True), timeout=60
    )

    assert served_during_check == [True, True]
    assert [type(result) for result in results[:2]] == [InvalidClientError] * 2


async def test_a_burst_of_secret_checks_leaves_other_thread_work_alone(
    mocker: pytest_mock.MockFixture, stored_app: OAuthApplicationInfoWithSecret
) -> None:
    """The checks queue on a pool of their own, so the threads
    asyncio.to_thread hands out, which API-key checks also use, stay free, and
    no more checks run at once than that pool has threads."""
    release = threading.Event()
    counter_lock = threading.Lock()
    running = 0
    most_at_once = 0

    def slow_check(*args: object) -> bool:
        nonlocal running, most_at_once
        with counter_lock:
            running += 1
            most_at_once = max(most_at_once, running)
        release.wait(timeout=60)
        with counter_lock:
            running -= 1
        return False

    mocker.patch.object(oauth.keysmith, "verify_key", side_effect=slow_check)
    # More checks than a default pool ever has threads: min(32, CPUs + 4).
    burst = asyncio.gather(
        *(validate_client_credentials("client-1", "agpt_wrong") for _ in range(40)),
        return_exceptions=True,
    )
    try:
        await _wait_for(lambda: running >= oauth.CLIENT_SECRET_CHECK_THREADS)
        other_work = await asyncio.wait_for(
            asyncio.to_thread(lambda: "done"), timeout=10
        )
    finally:
        release.set()
        results = await asyncio.wait_for(burst, timeout=60)

    assert other_work == "done"
    assert most_at_once == oauth.CLIENT_SECRET_CHECK_THREADS
    assert [type(result) for result in results] == [InvalidClientError] * 40


async def _wait_for(condition: Callable[[], bool]) -> None:
    for _ in range(3000):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within 30 s")


_NOT_ASCII = "\u00e9" * 43


@pytest.mark.parametrize(
    ("verifier", "challenge", "method"),
    [
        (_NOT_ASCII, "c" * 43, "S256"),
        (_NOT_ASCII, "c" * 43, "plain"),
        (_NOT_ASCII, "c" * 43, None),
        ("v" * 43, _NOT_ASCII, "S256"),
        (_NOT_ASCII, _NOT_ASCII, "plain"),
        ("\ud800" + "v" * 42, "c" * 43, "S256"),
    ],
    ids=[
        "verifier-s256",
        "verifier-plain",
        "verifier-no-method",
        "challenge",
        "both-and-equal",
        "lone-surrogate",
    ],
)
def test_a_verifier_or_challenge_that_is_not_ascii_never_matches(
    verifier: str, challenge: str, method: str | None
) -> None:
    """RFC 7636 verifiers and challenges are ASCII. Encoding one that is not
    raised UnicodeEncodeError under S256, and comparing it raised TypeError
    under plain."""
    assert oauth._verify_pkce(verifier, challenge, method) is False


@pytest.mark.parametrize(
    ("verifier", "challenge", "method", "matches"),
    [
        # The challenge is BASE64URL(SHA256("v" * 43)), without padding.
        ("v" * 43, "7w_YNF9DSfIdPf_pRjSq646_kPr-2-o9NAl16JGghdM", "S256", True),
        ("w" * 43, "7w_YNF9DSfIdPf_pRjSq646_kPr-2-o9NAl16JGghdM", "S256", False),
        ("v" * 43, "v" * 43, "plain", True),
        ("v" * 43, "v" * 43, None, True),
        ("v" * 43, "w" * 43, "plain", False),
        ("v" * 43, "v" * 43, "S512", False),
    ],
)
def test_an_ascii_verifier_is_compared_as_before(
    verifier: str, challenge: str, method: str | None, matches: bool
) -> None:
    assert oauth._verify_pkce(verifier, challenge, method) is matches
