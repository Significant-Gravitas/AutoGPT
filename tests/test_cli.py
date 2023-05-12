import pytest
from click.testing import CliRunner

from autogpt import cli
from autogpt.config.config import Config
from autogpt.configurator import GPT_3_MODEL, GPT_4_MODEL


@pytest.fixture(scope="function")
def mocked_config():
    global Config
    oldConfigClass = Config

    class MockConfig(oldConfigClass):
        pass

    yield
    Config = oldConfigClass


def test_gpt4only_cli_arg(mocked_config):
    runner = CliRunner()
    runner.invoke(cli.main, ["--gpt4only"])

    config = Config()
    assert config.smart_llm_model == GPT_4_MODEL
    assert config.fast_llm_model == GPT_4_MODEL


def test_gpt3only_cli_arg(mocked_config):
    runner = CliRunner()
    runner.invoke(cli.main, ["--gpt3only"])

    config = Config()
    assert config.smart_llm_model == GPT_3_MODEL
    assert config.fast_llm_model == GPT_3_MODEL
