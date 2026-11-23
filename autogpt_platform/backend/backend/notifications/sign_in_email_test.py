"""The sign-in emails: the code that confirms a provider sign-in's address and
the notice that a provider was added, both through the auth email kit."""

import re
from html import unescape

import pytest

from backend.notifications.auth import render_auth_email
from backend.notifications.renderer import build_urls

ADDRESS = "3rd Floor, 1 Ashley Road, Altrincham, WA14 2DT, UK"
CODE_INTRO = (
    "Someone is signing in to AutoGPT with Microsoft using this email address. "
    "Enter this code on the page that asked for it to finish signing in."
)
CODE_NOTE = (
    "If this wasn't you, ignore this email: nobody can sign in with Microsoft "
    "without this code. Don't share it with anyone."
)


def test_sign_in_code_email_shows_the_code_and_names_the_provider():
    email = render_auth_email(
        "sign_in_code", None, "sam@example.com", code="042917", provider="microsoft"
    )
    html = unescape(email.html)

    assert email.subject == "Your AutoGPT sign-in code"
    assert "ACCOUNT · SIGN-IN CODE" in html
    assert ">Confirm it's you<" in html
    assert f">{CODE_INTRO}</div>" in html
    assert ">042917</td>" in html
    assert ">The code expires in 10 minutes.</div>" in html
    assert f">{CODE_NOTE}</div>" in html
    assert "This code was requested by a Microsoft sign-in on AutoGPT." in html


def test_sign_in_code_email_has_no_button_or_link_to_follow():
    """The code is typed into the page that asked for it. A link would let the
    mailbox's owner approve someone else's sign-in with one click."""
    email = render_auth_email(
        "sign_in_code", None, "sam@example.com", code="042917", provider="google"
    )
    discord = build_urls("").discord

    assert 'bgcolor="#6144DF"' not in email.html
    assert "Copy this link" not in email.html
    assert set(re.findall(r'<a\b[^>]*?\bhref="([^"]+)"', email.html)) == {
        "https://agpt.co",
        "mailto:contact@agpt.co",
        discord,
    }


def test_sign_in_code_email_leaves_the_code_out_of_subject_and_preheader():
    """Both show on a locked phone's notifications."""
    email = render_auth_email(
        "sign_in_code", None, "sam@example.com", code="042917", provider="google"
    )

    assert "042917" not in email.subject
    assert "042917" not in email.preheader


def test_sign_in_code_plain_text_part():
    email = render_auth_email(
        "sign_in_code", None, "sam@example.com", code="042917", provider="microsoft"
    )
    discord = build_urls("").discord

    assert email.text == (
        "Confirm it's you\n"
        "\n"
        f"{CODE_INTRO}\n"
        "\n"
        "042917\n"
        "\n"
        "The code expires in 10 minutes.\n"
        "\n"
        f"{CODE_NOTE}\n"
        "\n"
        "This code was requested by a Microsoft sign-in on AutoGPT.\n"
        "Questions? Email contact@agpt.co\n"
        "\n"
        "This is a service message about an AutoGPT account. "
        "It's sent even if you've opted out of product news.\n"
        f"AutoGPT, {ADDRESS}\n"
        "https://agpt.co\n"
        f"Discord: {discord}"
    )


@pytest.mark.parametrize(
    "code, provider, message",
    [
        (None, "google", "needs a six-digit code"),
        ("12345", "google", "needs a six-digit code"),
        ("12345a", "google", "needs a six-digit code"),
        ("042917", None, "needs a provider"),
    ],
)
def test_a_sign_in_code_email_needs_a_code_and_a_provider(code, provider, message):
    with pytest.raises(ValueError, match=message):
        render_auth_email(
            "sign_in_code", None, "sam@example.com", code=code, provider=provider
        )


def test_sign_in_method_added_email_tells_the_owner_how_to_undo_it():
    email = render_auth_email(
        "sign_in_method_added", None, "sam@example.com", provider="github"
    )
    html = unescape(email.html)

    assert email.subject == "GitHub sign-in was added to your AutoGPT account"
    assert email.preheader == "If this was you, there's nothing to do."
    assert ">You can now sign in with GitHub<" in html
    assert (
        ">GitHub sign-in was just added to your AutoGPT account. If this was "
        "you, there's nothing to do. If it wasn't you, reply to this email and "
        "we'll remove it and secure your account.</div>"
    ) in html
    assert ">Questions? Email contact@agpt.co</a>" in html
    assert 'bgcolor="#6144DF"' not in email.html
    assert "expires" not in email.text


def test_a_sign_in_method_added_email_needs_a_provider():
    with pytest.raises(ValueError, match="needs a provider"):
        render_auth_email("sign_in_method_added", None, "sam@example.com")
