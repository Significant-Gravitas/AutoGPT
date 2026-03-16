from types import SimpleNamespace
from typing import Any, cast

from backend.api.test_helpers import override_config
from backend.notifications.email import EmailSender, settings
from backend.util.settings import AppEnvironment


def test_send_template_renders_nightly_copilot_email(mocker) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    sender.send_template(
        user_email="user@example.com",
        subject="Autopilot update",
        template_name="nightly_copilot.html.jinja2",
        data={
            "email_body_paragraphs": [
                "I found something useful for you.",
                "Open Copilot and I will walk you through it.",
            ],
            "cta_url": "https://example.com/copilot?callbackToken=token-1",
            "cta_label": "Open Copilot",
        },
    )

    body = send_email.call_args.kwargs["body"]

    assert "I found something useful for you." in body
    assert "Open Copilot" in body
    assert "Approval needed" not in body
    assert send_email.call_args.kwargs["user_unsubscribe_link"].endswith(
        "/profile/settings"
    )


def test_send_template_renders_nightly_copilot_approval_block(mocker) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    sender.send_template(
        user_email="user@example.com",
        subject="Autopilot update",
        template_name="nightly_copilot.html.jinja2",
        data={
            "email_body_paragraphs": ["I prepared a change worth reviewing."],
            "approval_summary_paragraphs": [
                "I drafted a follow-up because it matches your recent activity."
            ],
            "cta_url": "https://example.com/copilot?sessionId=session-1&showAutopilot=1",
            "cta_label": "Review in Copilot",
        },
    )

    body = send_email.call_args.kwargs["body"]

    assert "Approval needed" in body
    assert "If you want it to happen, please hit approve." in body
    assert "Review in Copilot" in body


def test_send_template_renders_nightly_copilot_callback_email(mocker) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    sender.send_template(
        user_email="user@example.com",
        subject="Autopilot update",
        template_name="nightly_copilot_callback.html.jinja2",
        data={
            "email_body_paragraphs": [
                "I prepared a follow-up based on your recent work."
            ],
            "cta_url": "https://example.com/copilot?callbackToken=token-1",
            "cta_label": "Open Copilot",
        },
    )

    body = send_email.call_args.kwargs["body"]

    assert "Autopilot picked up where you left off" in body
    assert "I prepared a follow-up based on your recent work." in body


def test_send_template_renders_nightly_copilot_callback_approval_block(mocker) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    sender.send_template(
        user_email="user@example.com",
        subject="Autopilot update",
        template_name="nightly_copilot_callback.html.jinja2",
        data={
            "email_body_paragraphs": [
                "I prepared a follow-up based on your recent work."
            ],
            "approval_summary_paragraphs": [
                "I want your approval before I apply the next step."
            ],
            "cta_url": "https://example.com/copilot?sessionId=session-1&showAutopilot=1",
            "cta_label": "Review in Copilot",
        },
    )

    body = send_email.call_args.kwargs["body"]

    assert "Approval needed" in body
    assert "I want your approval before I apply the next step." in body


def test_send_template_renders_nightly_copilot_invite_cta_email(mocker) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    sender.send_template(
        user_email="user@example.com",
        subject="Autopilot update",
        template_name="nightly_copilot_invite_cta.html.jinja2",
        data={
            "email_body_paragraphs": [
                "I put together an example of how Autopilot could help you."
            ],
            "cta_url": "https://example.com/copilot?callbackToken=token-1",
            "cta_label": "Try Copilot",
        },
    )

    body = send_email.call_args.kwargs["body"]

    assert "Your Autopilot beta access is waiting" in body
    assert "I put together an example of how Autopilot could help you." in body
    assert "Try Copilot" in body


def test_send_template_renders_nightly_copilot_invite_cta_approval_block(
    mocker,
) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    sender.send_template(
        user_email="user@example.com",
        subject="Autopilot update",
        template_name="nightly_copilot_invite_cta.html.jinja2",
        data={
            "email_body_paragraphs": [
                "I put together an example of how Autopilot could help you."
            ],
            "approval_summary_paragraphs": [
                "If this looks useful, approve the next step to try it."
            ],
            "cta_url": "https://example.com/copilot?sessionId=session-1&showAutopilot=1",
            "cta_label": "Review in Copilot",
        },
    )

    body = send_email.call_args.kwargs["body"]

    assert "Approval needed" in body
    assert "If this looks useful, approve the next step to try it." in body


def test_send_template_still_sends_in_production(mocker) -> None:
    sender = EmailSender()
    sender.postmark = cast(Any, object())
    send_email = mocker.patch.object(sender, "_send_email")

    with override_config(settings, "app_env", AppEnvironment.PRODUCTION):
        sender.send_template(
            user_email="user@example.com",
            subject="Autopilot update",
            template_name="nightly_copilot.html.jinja2",
            data={
                "email_body_paragraphs": ["I found something useful for you."],
                "cta_url": "https://example.com/copilot?callbackToken=token-1",
                "cta_label": "Open Copilot",
            },
        )

    send_email.assert_called_once()


def test_send_html_uses_default_unsubscribe_link(mocker) -> None:
    sender = EmailSender()
    send = mocker.Mock()
    sender.postmark = cast(Any, SimpleNamespace(emails=SimpleNamespace(send=send)))

    mocker.patch(
        "backend.notifications.email.get_frontend_base_url",
        return_value="https://example.com",
    )
    with override_config(settings, "postmark_sender_email", "test@example.com"):
        sender.send_html(
            user_email="user@example.com",
            subject="Autopilot update",
            body="<p>Hello</p>",
        )

    headers = send.call_args.kwargs["Headers"]

    assert headers["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    assert headers["List-Unsubscribe"] == "<https://example.com/profile/settings>"
