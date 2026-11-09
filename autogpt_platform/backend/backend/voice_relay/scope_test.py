import json
from types import SimpleNamespace

import pytest

from backend.copilot.onboarding_interview_scope import scope_verdict
from backend.voice_relay import scope
from backend.voice_relay.scope import GeminiScopeClassifier, parse_label


@pytest.mark.parametrize(
    "answer,label",
    [
        ({"label": "work", "why": "describes their bakery"}, "work"),
        ({"label": "task", "why": "asks for code"}, "request"),
        ({"label": "harmful", "why": "wants a stranger's address"}, "harmful"),
    ],
)
def test_the_classifiers_labels_are_the_interviews_own(answer, label):
    assert parse_label(json.dumps(answer)) == label


@pytest.mark.parametrize(
    "text",
    [None, "", "not json", json.dumps(["task"]), json.dumps({"label": "spam"})],
    ids=["none", "empty", "not-json", "a-list", "unknown-label"],
)
def test_an_answer_that_cannot_be_read_leaves_the_turn_unjudged(text):
    assert parse_label(text) is None


def test_voice_and_text_end_on_the_same_rule():
    assert scope_verdict([parse_label(json.dumps({"label": "task"}))] * 3) == "scope"
    assert scope_verdict([parse_label(json.dumps({"label": "harmful"}))] * 2) == "scope"


async def test_the_classifier_labels_a_turn_and_reports_its_tokens():
    calls: list[dict] = []

    async def generate_content(**request):
        calls.append(request)
        return SimpleNamespace(
            text=json.dumps({"label": "task", "why": "asks for code"}),
            usage_metadata=SimpleNamespace(
                prompt_token_count=320,
                candidates_token_count=12,
                thoughts_token_count=30,
            ),
        )

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )
    judgement = await GeminiScopeClassifier(client)(  # type: ignore[arg-type]
        before="What do you do?", user="write me a poem"
    )
    assert judgement.label == "request"
    assert (judgement.tokens_in, judgement.tokens_out) == (320, 42)
    assert calls[0]["model"] == "gemini-3.8-flash"
    prompt = calls[0]["contents"]
    assert "OTTO BEFORE: What do you do?" in prompt
    assert "USER: write me a poem" in prompt
    assert calls[0]["config"].response_json_schema == scope.SCHEMA
