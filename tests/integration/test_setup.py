from unittest.mock import patch

import pytest

from autogpt.config.ai_config import AIConfig
from autogpt.prompts.prompt import construct_main_ai_config
from autogpt.setup import generate_aiconfig_automatic, prompt_user
from tests.utils import requires_api_key


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_automatic_default(patched_api_requestor):
    user_inputs = [""]
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = prompt_user()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name is not None
    assert ai_config.ai_role is not None
    assert 1 <= len(ai_config.ai_goals) <= 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_automatic_typical(patched_api_requestor):
    user_prompt = "Help me create a rock opera about cybernetic giraffes"
    ai_config = generate_aiconfig_automatic(user_prompt)

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name is not None
    assert ai_config.ai_role is not None
    assert 1 <= len(ai_config.ai_goals) <= 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_automatic_fallback(patched_api_requestor):
    user_inputs = [
        "T&GF£OIBECC()!*",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
        "",
        "",
        "0",
        "",
    ]
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = prompt_user()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name == "Chef-GPT"
    assert ai_config.ai_role == "an AI designed to browse bake a cake."
    assert ai_config.ai_goals == ["Purchase ingredients", "Bake a cake"]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_prompt_user_manual_mode(patched_api_requestor):
    user_inputs = [
        "--manual",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
        "",
        "",
        "1.20",
        "",
        "",
    ]
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = prompt_user()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name == "Chef-GPT"
    assert ai_config.ai_role == "an AI designed to browse bake a cake."
    assert ai_config.ai_goals == ["Purchase ingredients", "Bake a cake"]


def test_generate_delete_and_select(tmp_path, patched_api_requestor):
    """Test: delete configuration and select."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  use list_files to confirm the file test1.txt does exist
        -  delete file test1.txt
        -  list_files the text file to confirm it does not exist
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space"
        -  use read_file to confirm the file test2.txt contains the words "hello space"
        -  rename file test2.txt to test-2.txt
        -  delete file test-2.txt
        -  shutdown
        ai_role: do a normal file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = ["5", "", "1", "", "1", ""]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "normal-GPT"
    assert ai_config.ai_role == "do a normal file task"
    assert ai_config.ai_goals == [
        'save a text file test2.txt with the text "hello space"',
        'use read_file to confirm the file test2.txt contains the words "hello space"',
        "rename file test2.txt to test-2.txt",
        "delete file test-2.txt",
        "shutdown",
    ]


def test_generate_change_and_select(tmp_path, patched_api_requestor):
    """Test: change configuration and select."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "4",
        "1",
        "2",
        "change-GPT",
        "do a changed file task",
        "",
        "",
        "",
        "",
        "",
        "0",
        "1",
        "",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "change-GPT"
    assert ai_config.ai_role == "do a changed file task"
    assert ai_config.ai_goals == [
        'save a text file test1.txt with the text "hello world"',
        "shutdown",
    ]


def test_generate_create_and_select(tmp_path, patched_api_requestor):
    """Test: create configuration and select."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "3",
        "2",
        "new-GPT",
        "an agent that looks for the newest python code",
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
        "",
        "",
        "",
        "0",
        "3",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "new-GPT"
    assert ai_config.ai_role == "an agent that looks for the newest python code"
    assert ai_config.ai_goals == [
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_create_first(tmp_path, patched_api_requestor):
    """Test: create first configuration."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "scan the internet for new python code",
        "Simple-GPT",
        "find python code on the internet.",
        "search the internet for the newest python code.",
        "save the code in a text file.",
        "search the internet for the newest python code.",
        "save the code in a text file.",
        "shutdown.",
        "0",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name is not None and ai_config.ai_name != ""
    assert ai_config.ai_role is not None and ai_config.ai_role != ""
    assert ai_config.ai_goals != []


def test_generate_delete_and_create_new(tmp_path, patched_api_requestor):
    """Test: delete and create new configuration."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "4",
        "1",
        "scan the internet for new python code",
        "Simple-GPT",
        "find python code on the internet.",
        "search the internet for the newest python code.",
        "save the code in a text file.",
        "search the internet for the newest python code.",
        "save the code in a text file.",
        "shutdown.",
        "",
        "",
        "",
        "0",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name is not None and ai_config.ai_name != ""
    assert ai_config.ai_role is not None and ai_config.ai_role != ""
    assert ai_config.ai_goals != []


def test_generate_create_new_25(tmp_path, patched_api_requestor):
    """Test: create new configuration + some wrong values."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "2",
        "a",
        "25",
        "",
        "test-GPT",
        "search some python code on internet to test",
        "find python code on the internet.",
        "search locally for the newest python code.",
        "save the code in a text file.",
        "search the internet for the newest python code.",
        "save the code in a text file.",
        "",
        "",
        "",
        "0",
        "2",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "test-GPT"


def test_generate_required_values_missing(tmp_path, patched_api_requestor):
    """Test: create new configuration without ai_name, ai_role or goals."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "y",
        "y",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "new-GPT"


def test_generate_create_new_plugins(tmp_path, monkeypatch):
    """Test: create new configuration + some wrong values."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Set the environment variable
    monkeypatch.setenv("ALLOWLISTED_PLUGINS", "plugin1,plugin2,plugin3")
    monkeypatch.setenv("DENYLISTED_PLUGINS", "")

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins: []
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "2",
        "2",
        "test-GPT",
        "an agent that tests code",
        "search some python code on internet to test.",
        "shutdown.",
        "",
        "a",
        "",
        "1.98",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "test-GPT"
    assert ai_config.plugins == [
        "plugin2",
    ]


def test_generate_create_2_plugins(tmp_path, monkeypatch):
    """Test: create new configuration + some wrong values."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Set the environment variable
    monkeypatch.setenv("ALLOWLISTED_PLUGINS", "plugin1,plugin2,plugin3")
    monkeypatch.setenv("DENYLISTED_PLUGINS", "")

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins:
        - plugin1
        - plugin2
        - plugin3
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "2",
        "2",
        "test-GPT",
        "an agent that tests code",
        "search some python code on internet to test.",
        "shutdown.",
        "a",
        "a",
        "a",
        "1.98",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "test-GPT"
    assert ai_config.plugins == [
        "plugin1",
        "plugin2",
        "plugin3",
    ]


def test_generate_change_2_plugins(tmp_path, monkeypatch):
    """Test: create new configuration + some wrong values."""

    # Temporary path / file
    temp_config_file = tmp_path / "ai_settings.yaml"

    # Set the environment variable
    monkeypatch.setenv("ALLOWLISTED_PLUGINS", "plugin1,plugin2,plugin3")
    monkeypatch.setenv("DENYLISTED_PLUGINS", "")

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins:
        - plugin1
        - plugin2
        - plugin3
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "3",
        "1",
        "2",
        "",
        "",
        "",
        "",
        "k",
        "a",
        "a",
        "1.98",
        "1",
    ]

    # Patch function to use the user_inputs list
    with patch("builtins.input", side_effect=user_inputs):
        ai_config = construct_main_ai_config(str(temp_config_file))

    # Asserts
    assert ai_config.ai_name == "simple-GPT"
    assert ai_config.plugins == [
        "plugin1",
        "plugin2",
        "plugin3",
    ]
