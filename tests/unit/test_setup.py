import pytest

from autogpt.config.ai_config import AIConfig
from autogpt.llm_utils import create_chat_completion
from autogpt.setup import generate_aiconfig_automatic, prompt_user
from tests.utils import requires_api_key


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_automatic_default(request, monkeypatch):
    user_inputs = [""]
    monkeypatch.setattr("builtins.input", lambda _: user_inputs.pop(0))
    ai_config = prompt_user()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name is not None
    assert ai_config.ai_role is not None
    assert 1 <= len(ai_config.ai_goals) <= 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_automatic_typical(request, mocker):
    user_prompt = "Help me create a rock opera about cybernetic giraffes"

    ai_config = generate_aiconfig_automatic(user_prompt)

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name is not None
    assert ai_config.ai_role is not None
    assert 1 <= len(ai_config.ai_goals) <= 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_automatic_fallback(request, monkeypatch):
    user_inputs = [
        "T&GF£OIBECC()!*",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
    ]
    monkeypatch.setattr("builtins.input", lambda _: user_inputs.pop(0))
    ai_config = prompt_user()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name == "Chef-GPT"
    assert ai_config.ai_role == "an AI designed to browse bake a cake."
    assert ai_config.ai_goals == ["Purchase ingredients", "Bake a cake"]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_prompt_user_manual_mode(request, monkeypatch):
    user_inputs = [
        "--manual",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
    ]
    monkeypatch.setattr("builtins.input", lambda _: user_inputs.pop(0))

    ai_config = prompt_user()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name == "Chef-GPT"
    assert ai_config.ai_role == "an AI designed to browse bake a cake."
    assert ai_config.ai_goals == ["Purchase ingredients", "Bake a cake"]
