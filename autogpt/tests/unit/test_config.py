"""
Test cases for the config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
import asyncio
import os
from typing import Any
from unittest import mock

import pytest
from forge.config.config import Config, ConfigBuilder
from forge.llm.providers.schema import ChatModelInfo, ModelProviderName
from openai.pagination import AsyncPage
from openai.types import Model
from pydantic import SecretStr

from autogpt.app.configurator import GPT_3_MODEL, GPT_4_MODEL, apply_overrides_to_config


def test_initial_values(config: Config) -> None:
    """
    Test if the initial values of the config class attributes are set correctly.
    """
    assert config.continuous_mode is False
    assert config.tts_config.speak_mode is False
    assert config.fast_llm.startswith("gpt-3.5-turbo")
    assert config.smart_llm.startswith("gpt-4")


@pytest.mark.asyncio
@mock.patch("openai.resources.models.AsyncModels.list")
async def test_fallback_to_gpt3_if_gpt4_not_available(
    mock_list_models: Any, config: Config
) -> None:
    """
    Test if models update to gpt-3.5-turbo if gpt-4 is not available.
    """
    config.fast_llm = GPT_4_MODEL
    config.smart_llm = GPT_4_MODEL

    mock_list_models.return_value = asyncio.Future()
    mock_list_models.return_value.set_result(
        AsyncPage(
            data=[Model(id=GPT_3_MODEL, created=0, object="model", owned_by="AutoGPT")],
            object="Models",  # no idea what this should be, but irrelevant
        )
    )

    await apply_overrides_to_config(
        config=config,
        gpt3only=False,
        gpt4only=False,
    )

    assert config.fast_llm == GPT_3_MODEL
    assert config.smart_llm == GPT_3_MODEL


def test_missing_azure_config(config: Config) -> None:
    assert config.openai_credentials is not None

    config_file = config.app_data_dir / "azure_config.yaml"
    with pytest.raises(FileNotFoundError):
        config.openai_credentials.load_azure_config(config_file)

    config_file.write_text("")
    with pytest.raises(ValueError):
        config.openai_credentials.load_azure_config(config_file)

    assert config.openai_credentials.api_type != "azure"
    assert config.openai_credentials.api_version == ""
    assert config.openai_credentials.azure_model_to_deploy_id_map is None


@pytest.fixture
def config_with_azure(config: Config):
    config_file = config.app_data_dir / "azure_config.yaml"
    config_file.write_text(
        f"""
azure_api_type: azure
azure_api_version: 2023-06-01-preview
azure_endpoint: https://dummy.openai.azure.com
azure_model_map:
    {config.fast_llm}: FAST-LLM_ID
    {config.smart_llm}: SMART-LLM_ID
    {config.embedding_model}: embedding-deployment-id-for-azure
"""
    )
    os.environ["USE_AZURE"] = "True"
    os.environ["AZURE_CONFIG_FILE"] = str(config_file)
    config_with_azure = ConfigBuilder.build_config_from_env(
        project_root=config.project_root
    )
    yield config_with_azure
    del os.environ["USE_AZURE"]
    del os.environ["AZURE_CONFIG_FILE"]


def test_azure_config(config_with_azure: Config) -> None:
    assert (credentials := config_with_azure.openai_credentials) is not None
    assert credentials.api_type == "azure"
    assert credentials.api_version == "2023-06-01-preview"
    assert credentials.azure_endpoint == SecretStr("https://dummy.openai.azure.com")
    assert credentials.azure_model_to_deploy_id_map == {
        config_with_azure.fast_llm: "FAST-LLM_ID",
        config_with_azure.smart_llm: "SMART-LLM_ID",
        config_with_azure.embedding_model: "embedding-deployment-id-for-azure",
    }

    fast_llm = config_with_azure.fast_llm
    smart_llm = config_with_azure.smart_llm
    assert (
        credentials.get_model_access_kwargs(config_with_azure.fast_llm)["model"]
        == "FAST-LLM_ID"
    )
    assert (
        credentials.get_model_access_kwargs(config_with_azure.smart_llm)["model"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt4only
    config_with_azure.fast_llm = smart_llm
    assert (
        credentials.get_model_access_kwargs(config_with_azure.fast_llm)["model"]
        == "SMART-LLM_ID"
    )
    assert (
        credentials.get_model_access_kwargs(config_with_azure.smart_llm)["model"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt3only
    config_with_azure.fast_llm = config_with_azure.smart_llm = fast_llm
    assert (
        credentials.get_model_access_kwargs(config_with_azure.fast_llm)["model"]
        == "FAST-LLM_ID"
    )
    assert (
        credentials.get_model_access_kwargs(config_with_azure.smart_llm)["model"]
        == "FAST-LLM_ID"
    )


@pytest.mark.asyncio
async def test_create_config_gpt4only(config: Config) -> None:
    with mock.patch(
        "forge.llm.providers.multi.MultiProvider.get_available_models"
    ) as mock_get_models:
        mock_get_models.return_value = [
            ChatModelInfo(
                name=GPT_4_MODEL,
                provider_name=ModelProviderName.OPENAI,
                max_tokens=4096,
            )
        ]
        await apply_overrides_to_config(
            config=config,
            gpt4only=True,
        )
        assert config.fast_llm == GPT_4_MODEL
        assert config.smart_llm == GPT_4_MODEL


@pytest.mark.asyncio
async def test_create_config_gpt3only(config: Config) -> None:
    with mock.patch(
        "forge.llm.providers.multi.MultiProvider.get_available_models"
    ) as mock_get_models:
        mock_get_models.return_value = [
            ChatModelInfo(
                name=GPT_3_MODEL,
                provider_name=ModelProviderName.OPENAI,
                max_tokens=4096,
            )
        ]
        await apply_overrides_to_config(
            config=config,
            gpt3only=True,
        )
        assert config.fast_llm == GPT_3_MODEL
        assert config.smart_llm == GPT_3_MODEL
