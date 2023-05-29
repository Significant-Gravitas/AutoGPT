""" Tests for the CLI."""
from unittest import mock

import pytest
from click.testing import CliRunner

from autogpt import cli
from autogpt.config.config import Config
from autogpt.configurator import GPT_3_MODEL, GPT_4_MODEL
from tests.utils import requires_api_key


@pytest.fixture
def mocked_config():
    """
    Fixture to mock the config object.
    """
    with mock.patch("autogpt.cli.Config") as mocked_config:
        yield mocked_config.return_value


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_gpt4only_cli_arg(patched_api_requestor) -> None:
    """
    Test that the --gpt4only CLI argument sets the smart and fast LLM models to GPT-4.
    """
    with mock.patch.object(Config, "set_smart_llm_model") as mocked_set_smart_llm_model:
        with mock.patch.object(
            Config, "set_fast_llm_model"
        ) as mocked_set_fast_llm_model:
            with mock.patch("openai.Model.list") as mock_list_models:
                # Test when correct model is returned
                mock_list_models.return_value = {"data": [{"id": GPT_4_MODEL}]}
                cli.main(gpt4only=True)
                mocked_set_smart_llm_model.assert_called_with(GPT_4_MODEL)
                mocked_set_fast_llm_model.assert_called_with(GPT_4_MODEL)


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_gpt3only_cli_arg(patched_api_requestor: None) -> None:
    """
    Test that the --gpt3only CLI argument sets the smart and fast LLM models to GPT-3.5.
    """
    with mock.patch.object(Config, "set_smart_llm_model") as mocked_set_smart_llm_model:
        with mock.patch.object(
            Config, "set_fast_llm_model"
        ) as mocked_set_fast_llm_model:
            runner = CliRunner()
            runner.invoke(cli.main, ["--gpt3only"])

            mocked_set_smart_llm_model.assert_called_with(GPT_3_MODEL)
            mocked_set_fast_llm_model.assert_called_with(GPT_3_MODEL)
