import pytest

from backend.voice_relay.language import spoken_language


@pytest.mark.parametrize(
    "text, language",
    [
        ("Got it. What does a normal week look like for you and your team?", "en"),
        ("Qué bien. ¿Y cómo es una semana normal para ti en el trabajo?", "es"),
        ("Legal. E como é uma semana normal para você no trabalho?", "pt"),
        ("D'accord. Et à quoi ressemble une semaine normale pour vous?", "fr"),
        ("Verstehe. Und wie sieht eine normale Woche für Sie aus, was ist das?", "de"),
        ("こんにちは、オットーです。お仕事は何をされていますか？", "ja"),
        ("你好，我是奥托。你是做什么工作的？", "zh"),
        ("안녕하세요, 저는 오토입니다. 무슨 일을 하세요?", "ko"),
        ("Привет, я Отто. Чем вы занимаетесь?", "ru"),
        ("Привіт, я Отто. Чим ви займаєтесь?", "uk"),
        ("नमस्ते, मैं ओटो हूँ। आप क्या काम करते हैं?", "hi"),
    ],
)
def test_a_turn_plainly_in_one_language_is_read_as_it(text, language):
    assert spoken_language(text) == language


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Okay.",
        "Sí.",
        "Mm-hm.",
        "Got it, Dana.",
        "12 3 4",
        # Too few of any language's common words to tell.
        "Perfect, bookkeeping then.",
    ],
)
def test_a_short_or_unclear_turn_is_not_read_as_any(text):
    assert spoken_language(text) is None
