from autogpt.config.ai_config import AIConfig

"""
Test cases for the AIConfig class, which handles loads the AI configuration
settings from a YAML file.
"""


def test_goals_are_always_lists_of_strings(tmp_path):
    """Test if the goals attribute is always a list of strings."""

    yaml_content = """
ai_goals:
- Goal 1: Make a sandwich
- Goal 2, Eat the sandwich
- Goal 3 - Go to sleep
- "Goal 4: Wake up"
ai_name: McFamished
ai_role: A hungry AI
api_budget: 0.0
"""
    config_file = tmp_path / "ai_settings.yaml"
    config_file.write_text(yaml_content)

    ai_config = AIConfig.load(config_file)

    assert len(ai_config.ai_goals) == 4
    assert ai_config.ai_goals[0] == "Goal 1: Make a sandwich"
    assert ai_config.ai_goals[1] == "Goal 2, Eat the sandwich"
    assert ai_config.ai_goals[2] == "Goal 3 - Go to sleep"
    assert ai_config.ai_goals[3] == "Goal 4: Wake up"

    config_file.write_text("")
    ai_config.save(config_file)

    yaml_content2 = """ai_goals:
- 'Goal 1: Make a sandwich'
- Goal 2, Eat the sandwich
- Goal 3 - Go to sleep
- 'Goal 4: Wake up'
ai_name: McFamished
ai_role: A hungry AI
api_budget: 0.0
"""
    assert config_file.read_text() == yaml_content2


def test_ai_config_file_not_exists(workspace):
    """Test if file does not exist."""

    config_file = workspace.get_path("ai_settings.yaml")

    ai_config = AIConfig.load(str(config_file))
    assert ai_config.ai_name == ""
    assert ai_config.ai_role == ""
    assert ai_config.ai_goals == []
    assert ai_config.api_budget == 0.0
    assert ai_config.prompt_generator is None
    assert ai_config.command_registry is None


def test_ai_config_file_is_empty(workspace):
    """Test if file does not exist."""

    config_file = workspace.get_path("ai_settings.yaml")
    config_file.write_text("")

    ai_config = AIConfig.load(str(config_file))
    assert ai_config.ai_name == ""
    assert ai_config.ai_role == ""
    assert ai_config.ai_goals == []
    assert ai_config.api_budget == 0.0
    assert ai_config.prompt_generator is None
    assert ai_config.command_registry is None
