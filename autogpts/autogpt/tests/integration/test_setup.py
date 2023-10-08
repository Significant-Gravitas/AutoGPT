from unittest.mock import patch

import pytest

from autogpt.app.setup import generate_aiconfig_automatic, interactive_ai_profile_setup
from autogpt.config.ai_profile import AIProfile


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
async def test_generate_aiconfig_automatic_default(
    patched_api_requestor, config, llm_provider
):
    user_inputs = [""]
    with patch("autogpt.app.utils.session.prompt", side_effect=user_inputs):
        ai_profile = await interactive_ai_profile_setup(config, llm_provider)

    assert isinstance(ai_profile, AIProfile)
    assert ai_profile.ai_name is not None
    assert ai_profile.ai_role is not None
    assert 1 <= len(ai_profile.ai_goals) <= 5


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
async def test_generate_aiconfig_automatic_typical(
    patched_api_requestor, config, llm_provider
):
    user_prompt = "Help me create a rock opera about cybernetic giraffes"
    ai_profile = await generate_aiconfig_automatic(user_prompt, config, llm_provider)

    assert isinstance(ai_profile, AIProfile)
    assert ai_profile.ai_name is not None
    assert ai_profile.ai_role is not None
    assert 1 <= len(ai_profile.ai_goals) <= 5


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
async def test_generate_aiconfig_automatic_fallback(
    patched_api_requestor, config, llm_provider
):
    user_inputs = [
        "T&GF£OIBECC()!*",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
    ]
    with patch("autogpt.app.utils.session.prompt", side_effect=user_inputs):
        ai_profile = await interactive_ai_profile_setup(config, llm_provider)

    assert isinstance(ai_profile, AIProfile)
    assert ai_profile.ai_name == "Chef-GPT"
    assert ai_profile.ai_role == "an AI designed to browse bake a cake."
    assert ai_profile.ai_goals == ["Purchase ingredients", "Bake a cake"]


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
async def test_prompt_user_manual_mode(patched_api_requestor, config, llm_provider):
    user_inputs = [
        "--manual",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
    ]
    with patch("autogpt.app.utils.session.prompt", side_effect=user_inputs):
        ai_profile = await interactive_ai_profile_setup(config, llm_provider)

    assert isinstance(ai_profile, AIProfile)
    assert ai_profile.ai_name == "Chef-GPT"
    assert ai_profile.ai_role == "an AI designed to browse bake a cake."
    assert ai_profile.ai_goals == ["Purchase ingredients", "Bake a cake"]
