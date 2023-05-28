""" Tests for the CLI."""
from unittest import mock

from click.testing import CliRunner

from autogpt import cli
from autogpt.config.config import Config
from autogpt.configurator import GPT_3_MODEL, GPT_4_MODEL


def test_gpt4only_cli_arg() -> None:
    """
    Test that the --gpt4only CLI argument sets the smart and fast LLM models to GPT-4.
    """
    with mock.patch.object(Config, "set_smart_llm_model") as mocked_set_smart_llm_model:
        with mock.patch.object(
            Config, "set_fast_llm_model"
        ) as mocked_set_fast_llm_model:
            runner = CliRunner()
            runner.invoke(cli.main, ["--gpt4only"])
            mocked_set_smart_llm_model.assert_called_with(GPT_4_MODEL)
            mocked_set_fast_llm_model.assert_called_with(GPT_4_MODEL)


def test_gpt3only_cli_arg() -> None:
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
