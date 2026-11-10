import pytest

from .options import TurnOptions, parse_turn_options


@pytest.mark.parametrize(
    "text,expert,model,rest",
    [
        ("expert=Bolt fix the build", "Bolt", None, "fix the build"),
        ('expert="Bolt the Builder" hi', "Bolt the Builder", None, "hi"),
        ("expert='Bolt the Builder' hi", "Bolt the Builder", None, "hi"),
        ("mode=smart why?", None, "advanced", "why?"),
        ("mode=fast why?", None, "standard", "why?"),
        ("mode=advanced why?", None, "advanced", "why?"),
        ("mode=standard why?", None, "standard", "why?"),
        ("EXPERT=bolt MODE=Smart hi", "bolt", "advanced", "hi"),
        ("mode=smart expert=Bolt hi", "Bolt", "advanced", "hi"),
        ("  expert=Bolt   hi", "Bolt", None, "hi"),
        ("expert=Bolt", "Bolt", None, ""),
    ],
)
def test_leading_options_are_read(text, expert, model, rest):
    options, cleaned = parse_turn_options(text)
    assert options == TurnOptions(expert_name=expert, model=model)
    assert cleaned == rest


@pytest.mark.parametrize(
    "text",
    [
        "hello expert=Bolt",
        "please use mode=smart",
        "color=red expert=Bolt hi",
        "mode=turbo hi",
        'expert="" hi',
        "expertise=high hi",
        "",
    ],
)
def test_non_leading_unknown_or_invalid_tokens_stay_text(text):
    options, cleaned = parse_turn_options(text)
    assert options == TurnOptions()
    assert cleaned is text


def test_unknown_key_after_known_stops_parsing():
    options, cleaned = parse_turn_options("mode=fast color=red hi")
    assert options == TurnOptions(model="standard")
    assert cleaned == "color=red hi"


def test_merged_prefers_newer_and_keeps_unset():
    old = TurnOptions(expert_name="Bolt", expert_id="e1", model="advanced")
    assert old.merged(TurnOptions(model="standard")) == TurnOptions(
        expert_name="Bolt", expert_id="e1", model="standard"
    )
    assert old.merged(TurnOptions()) == old
