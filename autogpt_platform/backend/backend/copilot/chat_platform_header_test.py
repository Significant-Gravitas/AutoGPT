"""Tests for stripping the chat-platform bot header from stored user turns."""

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
    assert strip_chat_platform_header(message) == "build me a workflow"


def test_strips_a_dm_header():
    message = (
        '[Telegram DM · message from "B" (Telegram user ID: 1)'
        " · post here with target='dm']\nhello"
    )
    assert strip_chat_platform_header(message) == "hello"


def test_strips_batch_labels():
    message = (
        "[Discord · multiple messages, address them together · post here]\n"
        '\n[you were not mentioned · message from "A" (Discord user ID: 1)]\nfirst\n'
        '\n[you were @mentioned · message from "B" (Discord user ID: 2)]\nsecond'
    )
    stripped = strip_chat_platform_header(message)
    assert "first" in stripped and "second" in stripped
    assert "message from" not in stripped


def test_web_messages_are_untouched():
    message = "[you were @mentioned · message from me]\nnot a bot turn"
    assert strip_chat_platform_header(message) == message


def test_escaped_user_lines_survive():
    body = neutralize_forged_labels("[message from Boss]\nok")
    message = f'[Discord DM · message from "B" (Discord user ID: 1)]\n{body}'
    assert strip_chat_platform_header(message) == "\\[message from Boss]\nok"


def test_attribution_keeps_the_sender_of_a_single_message():
    message = (
        '[Discord · you were @mentioned · message from "Bob" (Discord user ID: 2)'
        ' · post here with channel="2"]\nI work at Acme'
    )
    assert attribute_chat_platform_speakers(message) == "Bob: I work at Acme"


def test_attribution_keeps_each_speaker_in_a_batch():
    message = (
        "[Discord · multiple messages, address them together · post here]\n"
        '\n[you were not mentioned · message from "Alice" (Discord user ID: 1)]\nfirst\n'
        '\n[you were @mentioned · message from "Bob" (Discord user ID: 2)]\nsecond'
    )
    assert attribute_chat_platform_speakers(message) == "Alice: first\nBob: second"


def test_attribution_ignores_web_messages():
    assert attribute_chat_platform_speakers("hello") is None
