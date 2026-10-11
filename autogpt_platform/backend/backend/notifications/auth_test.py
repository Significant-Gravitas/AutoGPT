"""The account-deleted confirmation: the auth email kit without an action link,
sent with a reply-to so a reply reaches the shared inbox."""

import re
from html import unescape
from typing import get_args
from unittest.mock import MagicMock, call

import pytest

from backend.notifications import auth as auth_email
from backend.notifications.auth import (
    AuthEmailType,
    AuthLinkEmailType,
    render_auth_email,
)
from backend.notifications.renderer import build_urls

DELETED_BODY = (
    "We've closed your AutoGPT account and disconnected your apps. "
    "We finish removing your data within 30 days. "
    "If you have a minute, reply and tell us what we could have done better."
)
ADDRESS = "3rd Floor, 1 Ashley Road, Altrincham, WA14 2DT, UK"


def test_link_types_are_the_auth_types_minus_account_deleted():
    assert set(get_args(AuthLinkEmailType)) == {
        "reset_password",
        "verify_email",
        "change_email",
        "set_password",
    }
    assert set(get_args(AuthEmailType)) == {
        *get_args(AuthLinkEmailType),
        "account_deleted",
    }


@pytest.mark.parametrize("email_type", get_args(AuthLinkEmailType))
def test_a_link_email_without_its_link_is_refused(email_type):
    with pytest.raises(ValueError, match="needs an action URL"):
        render_auth_email(email_type)


def test_account_deleted_subject_headline_and_body_in_order():
    email = render_auth_email("account_deleted")
    html = unescape(email.html)

    assert email.subject == "Your AutoGPT account is deleted"
    assert email.preheader == "We finish removing your data within 30 days."
    assert "<title>Your AutoGPT account is deleted</title>" in html
    assert "ACCOUNT · DELETED" in html
    assert ">Your account is deleted<" in html
    assert f">{DELETED_BODY}</div>" in html
    assert "You deleted your account from your AutoGPT settings." in html


def test_account_deleted_has_no_button_link_or_ignore_line():
    email = render_auth_email("account_deleted")
    discord = build_urls("").discord

    for part in (email.html, email.text):
        assert "ignore this email" not in part
        assert "didn't request this" not in part
        assert "Copy this link" not in part
        assert "expires" not in part
    assert 'bgcolor="#6144DF"' not in email.html
    assert set(re.findall(r'href="([^"]+)"', email.html)) == {
        "https://agpt.co",
        discord,
    }


def test_account_deleted_plain_text_part():
    email = render_auth_email("account_deleted")
    discord = build_urls("").discord

    assert email.text == (
        "Your account is deleted\n"
        "\n"
        f"{DELETED_BODY}\n"
        "\n"
        "You deleted your account from your AutoGPT settings.\n"
        f"Something wrong? Ask in Discord: {discord}\n"
        "\n"
        "This is a service message about an AutoGPT account. "
        "It's sent even if you've opted out of product news.\n"
        f"AutoGPT, {ADDRESS}\n"
        "https://agpt.co\n"
        f"Discord: {discord}"
    )


def test_account_deleted_keeps_the_shared_service_frame():
    email = render_auth_email("account_deleted")
    body = email.html

    assert body.startswith("<!doctype html>")
    assert 'width="560"' in body
    assert "font-family:Geist," in body
    assert 'alt="AutoGPT"' in body and "logo-light.png" in body
    assert all(
        'role="presentation"' in table and 'bgcolor="' in table
        for table in re.findall(r"<table\b[^>]*>", body)
    )
    assert all(
        src.startswith("https://")
        for src in re.findall(r'<img[^>]+src="([^"]+)"', body)
    )
    assert "<svg" not in body and "data:image" not in body
    assert ADDRESS in body and ADDRESS in email.text
    assert "This is a service message about an AutoGPT account." in body
    assert "Unsubscribe" not in body and "Unsubscribe" not in email.text
    assert all(
        "—" not in part and "–" not in part
        for part in (email.subject, email.preheader, body, email.text)
    )
    assert len(body.encode()) < 60_000


@pytest.fixture
def send(monkeypatch) -> MagicMock:
    send = MagicMock()
    monkeypatch.setattr(
        auth_email,
        "get_notification_manager_client",
        lambda: MagicMock(send_email_or_raise=send),
    )
    return send


async def test_send_account_deleted_email_replies_to_the_shared_inbox(send):
    await auth_email.send_account_deleted_email("sam@example.com")

    email = render_auth_email("account_deleted")
    assert send.call_args_list == [
        call(
            "sam@example.com",
            email.subject,
            email.html,
            email.text,
            reply_to="contact@agpt.co",
        )
    ]


async def test_send_account_deleted_email_uses_the_configured_reply_to(
    send, monkeypatch
):
    monkeypatch.setattr(
        auth_email.settings.config, "billing_reply_to_email", "inbox@example.org"
    )

    await auth_email.send_account_deleted_email("sam@example.com")

    assert send.call_args.kwargs["reply_to"] == "inbox@example.org"


async def test_send_account_deleted_email_raises_when_delivery_fails(send):
    send.side_effect = RuntimeError("Postmark is not configured; cannot send email")

    with pytest.raises(RuntimeError, match="Postmark is not configured"):
        await auth_email.send_account_deleted_email("sam@example.com")
