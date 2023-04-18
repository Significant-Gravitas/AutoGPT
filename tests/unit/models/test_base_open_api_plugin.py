import pytest
from typing import Any, Dict, List, Optional, Tuple
from autogpt.models.base_open_ai_plugin import BaseOpenAIPlugin, Message, PromptGenerator


class DummyPlugin(BaseOpenAIPlugin):
    pass


@pytest.fixture
def dummy_plugin():
    manifests_specs_clients = {
        "manifest": {
            "name_for_model": "Dummy",
            "schema_version": "1.0",
            "description_for_model": "A dummy plugin for testing purposes"
        },
        "client": None,
        "openapi_spec": None
    }
    return DummyPlugin(manifests_specs_clients)


def test_dummy_plugin_inheritance(dummy_plugin):
    assert isinstance(dummy_plugin, BaseOpenAIPlugin)


def test_dummy_plugin_name(dummy_plugin):
    assert dummy_plugin._name == "Dummy"


def test_dummy_plugin_version(dummy_plugin):
    assert dummy_plugin._version == "1.0"


def test_dummy_plugin_description(dummy_plugin):
    assert dummy_plugin._description == "A dummy plugin for testing purposes"


def test_dummy_plugin_default_methods(dummy_plugin):
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

    assert dummy_plugin.on_response(None) is None
    assert dummy_plugin.post_prompt(None) is None
    assert dummy_plugin.on_planning(None, None) is None
    assert dummy_plugin.post_planning(None) is None
    assert dummy_plugin.pre_instruction(None) is None
    assert dummy_plugin.on_instruction(None) is None
    assert dummy_plugin.post_instruction(None) is None
    assert dummy_plugin.pre_command(None, None) is None
    assert dummy_plugin.post_command(None, None) is None
    assert dummy_plugin.handle_chat_completion(None, None, None, None) is None
