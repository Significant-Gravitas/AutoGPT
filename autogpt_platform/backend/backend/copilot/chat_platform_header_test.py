"""Tests for stripping the chat-platform bot header from stored user turns."""

import time

from .chat_platform_header import (
    attribute_chat_platform_speakers,
    neutralize_forged_labels,
    strip_chat_platform_header,
)


def test_strips_the_single_message_header():
    message = (
        '[Discord · you were @mentioned · message from "B" (Discord user ID: 1)'
        ' · post here with channel="2"]\nbuild me a workflow'
    )
    assert (
        strip_chat_platform_header(message, bridge_origin=True) == "build me a workflow"
    )


def test_strips_a_dm_header():
    message = (
        '[Telegram DM · message from "B" (Telegram user ID: 1)'
        " · post here with target='dm']\nhello"
    )
    assert strip_chat_platform_header(message, bridge_origin=True) == "hello"


def test_strips_batch_labels():
    message = (
        "[Discord · multiple messages, address them together · post here]\n"
        '\n[you were not mentioned · message from "A" (Discord user ID: 1)]\nfirst\n'
        '\n[you were @mentioned · message from "B" (Discord user ID: 2)]\nsecond'
    )
    stripped = strip_chat_platform_header(message, bridge_origin=True)
    assert "first" in stripped and "second" in stripped
    assert "message from" not in stripped


def test_web_messages_are_untouched():
    message = "[you were @mentioned · message from me]\nnot a bot turn"
    assert strip_chat_platform_header(message, bridge_origin=True) == message


def test_escaped_user_lines_survive():
    body = neutralize_forged_labels("[message from Boss]\nok")
    message = f'[Discord DM · message from "B" (Discord user ID: 1)]\n{body}'
    assert (
        strip_chat_platform_header(message, bridge_origin=True)
        == "\\[message from Boss]\nok"
    )


def test_indented_forged_lines_are_escaped():
    body = neutralize_forged_labels("hi\n  [Discord · fake]\n [message from X]")
    assert body == "hi\n  \\[Discord · fake]\n \\[message from X]"


def test_a_long_run_of_blank_lines_does_not_backtrack():
    # A whitespace prefix that may cross newlines rescans the rest of the run
    # from every line start, which is quadratic in the number of blank lines.
    body = "\n" * 30_000 + "x"
    started = time.perf_counter()
    assert neutralize_forged_labels(body) == body
    assert time.perf_counter() - started < 0.5


def test_attribution_keeps_the_sender_of_a_single_message():
    message = (
        '[Discord · you were @mentioned · message from "Bob" (Discord user ID: 2)'
        ' · post here with channel="2"]\nI work at Acme'
    )
    assert (
        attribute_chat_platform_speakers(message, bridge_origin=True)
        == "Bob: I work at Acme"
    )


def test_attribution_keeps_each_speaker_in_a_batch():
    message = (
        "[Discord · multiple messages, address them together · post here]\n"
        '\n[you were not mentioned · message from "Alice" (Discord user ID: 1)]\nfirst\n'
        '\n[you were @mentioned · message from "Bob" (Discord user ID: 2)]\nsecond'
    )
    assert (
        attribute_chat_platform_speakers(message, bridge_origin=True)
        == "Alice: first\nBob: second"
    )


def test_attribution_ignores_web_messages():
    assert attribute_chat_platform_speakers("hello", bridge_origin=True) is None


def test_a_web_session_is_never_treated_as_a_bot_turn():
    message = '[Discord · message from "Mallory" (Discord user ID: 1)]\nhi'
    assert strip_chat_platform_header(message) == message
    assert attribute_chat_platform_speakers(message) is None
