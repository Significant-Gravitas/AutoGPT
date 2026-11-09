import base64
import json
import time

import pytest

from backend.voice_relay import tokens
from backend.voice_relay.tokens import TokenError

USER = "user-1"


def _call_id() -> str:
    return tokens.new_call_id()


def test_a_start_token_carries_its_user_call_and_leg():
    call_id = _call_id()
    claims = tokens.verify_start(tokens.mint_start(USER, call_id, "websocket"))
    assert (claims.user_id, claims.call_id, claims.leg) == (USER, call_id, "websocket")
    assert claims.expires_at == pytest.approx(time.time() + tokens.TOKEN_TTL_S, abs=2)


def test_a_token_signed_with_another_key_is_refused(monkeypatch):
    token = tokens.mint_start(USER, _call_id(), "websocket")
    monkeypatch.setattr(tokens.settings.secrets, "voice_relay_signing_key", "z" * 48)
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_start(token)


def test_a_token_past_its_expiry_is_refused():
    token = tokens.mint_start(USER, _call_id(), "websocket", now=time.time() - 120)
    with pytest.raises(TokenError, match="token_expired"):
        tokens.verify_start(token)


def test_a_tampered_claim_breaks_the_signature():
    token = tokens.mint_start(USER, _call_id(), "websocket")
    body, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    claims["uid"] = "someone-else"
    forged = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_start(f"{forged.decode()}.{signature}")


@pytest.mark.parametrize(
    "token",
    ["", "no-dot", "a.b.c", "x" * 2_000, "body.sïgnature", None, 42],
    ids=["empty", "no-dot", "three-parts", "huge", "non-ascii", "none", "number"],
)
def test_garbage_is_refused_as_an_invalid_token(token):
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_start(token)


def test_a_resume_ticket_does_not_start_a_call():
    ticket = tokens.mint_resume(USER, _call_id(), ttl_s=300)
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_start(ticket)


def test_a_start_token_does_not_resume_a_call():
    token = tokens.mint_start(USER, _call_id(), "websocket")
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_resume(token)


def test_a_resume_ticket_names_its_user_and_call():
    call_id = _call_id()
    claims = tokens.verify_resume(tokens.mint_resume(USER, call_id, ttl_s=360))
    assert (claims.user_id, claims.call_id) == (USER, call_id)


def test_a_token_naming_an_unknown_leg_is_refused():
    token = tokens._sign(
        {
            "kind": "start",
            "uid": USER,
            "call_id": _call_id(),
            "leg": "carrier-pigeon",
            "exp": time.time() + 60,
        }
    )
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_start(token)


def test_a_call_id_that_could_reach_another_key_is_refused():
    token = tokens._sign(
        {
            "kind": "start",
            "uid": USER,
            "call_id": "abc:other",
            "leg": "websocket",
            "exp": time.time() + 60,
        }
    )
    with pytest.raises(TokenError, match="token_invalid"):
        tokens.verify_start(token)


@pytest.mark.parametrize("key", ["", "short-key"], ids=["blank", "too-short"])
def test_no_token_is_minted_or_trusted_without_a_proper_key(monkeypatch, key):
    token = tokens.mint_start(USER, _call_id(), "websocket")
    monkeypatch.setattr(tokens.settings.secrets, "voice_relay_signing_key", key)
    assert not tokens.is_configured()
    with pytest.raises(TokenError, match="not_configured"):
        tokens.mint_start(USER, _call_id(), "websocket")
    with pytest.raises(TokenError, match="not_configured"):
        tokens.verify_start(token)


async def test_a_start_token_is_spent_by_its_first_use(redis):
    claims = tokens.verify_start(tokens.mint_start(USER, _call_id(), "websocket"))
    await tokens.spend(redis, claims)
    assert 0 < await redis.ttl(tokens.spent_key(claims.call_id)) <= 90
    with pytest.raises(TokenError, match="token_used"):
        await tokens.spend(redis, claims)


def test_a_start_token_carries_the_persons_ending_arm():
    token = tokens.mint_start(USER, _call_id(), "websocket", ending="02-profile-card")
    assert tokens.verify_start(token).ending == "02-profile-card"
    assert (
        tokens.verify_start(tokens.mint_start(USER, _call_id(), "websocket")).ending
        is None
    )


@pytest.mark.parametrize("ending", [7, "", "02 profile card", "x" * 65])
def test_a_token_naming_an_arm_that_is_not_one_starts_an_unenrolled_call(ending):
    """The arm is the only claim a call can do without: the call goes ahead,
    ending as an unenrolled one does."""
    call_id = _call_id()
    token = tokens._sign(
        {
            "kind": "start",
            "uid": USER,
            "call_id": call_id,
            "leg": "websocket",
            "exp": time.time() + 60,
            "ending": ending,
        }
    )

    claims = tokens.verify_start(token)

    assert (claims.user_id, claims.call_id, claims.ending) == (USER, call_id, None)
