import pytest

from backend.copilot.sdk.model_gateway.adapters.codex_signature import (
    reasoning_tag,
    seal,
    unseal,
)

_ENCRYPTED = "gAAAAABo" + "x" * 4096


def test_a_sealed_signature_opens_under_the_same_account_and_model():
    tag = reasoning_tag("acct-1", "gpt-5.6-terra")
    assert unseal(seal(_ENCRYPTED, tag), tag) == _ENCRYPTED


def test_encrypted_content_with_dots_survives():
    tag = reasoning_tag("acct-1", "m")
    assert unseal(seal("a.b.c", tag), tag) == "a.b.c"


@pytest.mark.parametrize(
    "account, model",
    [("acct-2", "gpt-5.6-terra"), ("acct-1", "gpt-5.5")],
    ids=["other-account", "other-model"],
)
def test_reasoning_from_another_account_or_model_is_not_sent_back(account, model):
    signature = seal(_ENCRYPTED, reasoning_tag("acct-1", "gpt-5.6-terra"))
    assert unseal(signature, reasoning_tag(account, model)) is None


@pytest.mark.parametrize(
    "signature",
    [
        "",
        None,
        "EqQBCkYIBRgCIkDanthropic-signature",
        "agptcx1.only-a-tag",
        "agptcx2.t.x",
    ],
)
def test_signatures_from_anything_else_are_dropped(signature):
    assert unseal(signature, reasoning_tag("acct-1", "m")) is None


def test_the_signature_reveals_neither_account_nor_model():
    signature = seal("encrypted", reasoning_tag("acct-secret-123", "model"))
    assert len(signature.split(".")[1]) == 64
    assert "acct-secret-123" not in signature and "model" not in signature
