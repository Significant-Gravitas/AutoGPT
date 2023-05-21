from typing import Any, Dict, List, Optional, Tuple

import pytest

from autogpt.models.base_open_ai_plugin import (
    BaseOpenAIPlugin,
    Message,
    PromptGenerator,
)


class DummyPlugin(BaseOpenAIPlugin):
    """A dummy plugin for testing purposes."""

    pass


@pytest.fixture
def dummy_plugin():
    """A dummy plugin for testing purposes."""
    manifests_specs_clients = {
        "manifest": {
            "name_for_model": "Dummy",
            "schema_version": "1.0",
            "description_for_model": "A dummy plugin for testing purposes",
        },
        "client": None,
        "openapi_spec": None,
    }
    return DummyPlugin(manifests_specs_clients)


def test_dummy_plugin_inheritance(dummy_plugin):
    """Test that the DummyPlugin class inherits from the BaseOpenAIPlugin class."""
    assert isinstance(dummy_plugin, BaseOpenAIPlugin)


def test_dummy_plugin_name(dummy_plugin):
    """Test that the DummyPlugin class has the correct name."""
    assert dummy_plugin._name == "Dummy"


def test_dummy_plugin_version(dummy_plugin):
    """Test that the DummyPlugin class has the correct version."""
    assert dummy_plugin._version == "1.0"


def test_dummy_plugin_description(dummy_plugin):
    """Test that the DummyPlugin class has the correct description."""
    assert dummy_plugin._description == "A dummy plugin for testing purposes"


def test_dummy_plugin_default_methods(dummy_plugin):
    """Test that the DummyPlugin class has the correct default methods."""
    assert not dummy_plugin.can_handle_on_response()
    assert not dummy_plugin.can_handle_post_prompt()
    assert not dummy_plugin.can_handle_on_planning()
    assert not dummy_plugin.can_handle_post_planning()
    assert not dummy_plugin.can_handle_pre_instruction()
    assert not dummy_plugin.can_handle_on_instruction()
    assert not dummy_plugin.can_handle_post_instruction()
    assert not dummy_plugin.can_handle_pre_command()
    assert not dummy_plugin.can_handle_post_command()
    assert not dummy_plugin.can_handle_chat_completion(None, None, None, None)

    assert dummy_plugin.on_response("hello") == "hello"
    assert dummy_plugin.post_prompt(None) is None
    assert dummy_plugin.on_planning(None, None) is None
    assert dummy_plugin.post_planning("world") == "world"
    pre_instruction = dummy_plugin.pre_instruction(
        [{"role": "system", "content": "Beep, bop, boop"}]
    )
    assert isinstance(pre_instruction, list)
    assert len(pre_instruction) == 1
    assert pre_instruction[0]["role"] == "system"
    assert pre_instruction[0]["content"] == "Beep, bop, boop"
    assert dummy_plugin.on_instruction(None) is None
    assert dummy_plugin.post_instruction("I'm a robot") == "I'm a robot"
    pre_command = dummy_plugin.pre_command("evolve", {"continuously": True})
    assert isinstance(pre_command, tuple)
    assert len(pre_command) == 2
    assert pre_command[0] == "evolve"
    assert pre_command[1]["continuously"] == True
    post_command = dummy_plugin.post_command("evolve", "upgraded successfully!")
    assert isinstance(post_command, str)
    assert post_command == "upgraded successfully!"
    assert dummy_plugin.handle_chat_completion(None, None, None, None) is None
