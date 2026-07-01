from types import SimpleNamespace

from backend.util.llm.conversions import extract_openai_reasoning


def test_extracts_openai_compatible_reasoning_content() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(reasoning_content="ok", content=""))
        ]
    )

    assert extract_openai_reasoning(response) == "ok"
