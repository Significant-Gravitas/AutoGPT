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
    ai_settings_file = tmp_path / "ai_settings.yaml"
    ai_settings_file.write_text(yaml_content)

    ai_config = AIConfig.load(ai_settings_file)

    assert len(ai_config.ai_goals) == 4
    assert ai_config.ai_goals[0] == "Goal 1: Make a sandwich"
    assert ai_config.ai_goals[1] == "Goal 2, Eat the sandwich"
    assert ai_config.ai_goals[2] == "Goal 3 - Go to sleep"
    assert ai_config.ai_goals[3] == "Goal 4: Wake up"

    ai_settings_file.write_text("")
    ai_config.save(ai_settings_file)

    yaml_content2 = """ai_goals:
- 'Goal 1: Make a sandwich'
- Goal 2, Eat the sandwich
- Goal 3 - Go to sleep
- 'Goal 4: Wake up'
ai_name: McFamished
ai_role: A hungry AI
api_budget: 0.0
"""
    assert ai_settings_file.read_text() == yaml_content2


def test_ai_config_file_not_exists(workspace):
    """Test if file does not exist."""

    ai_settings_file = workspace.get_path("ai_settings.yaml")

    ai_config = AIConfig.load(str(ai_settings_file))
    assert ai_config.ai_name == ""
    assert ai_config.ai_role == ""
    assert ai_config.ai_goals == []
    assert ai_config.api_budget == 0.0
    assert ai_config.prompt_generator is None
    assert ai_config.command_registry is None


def test_ai_config_file_is_empty(workspace):
    """Test if file does not exist."""

    ai_settings_file = workspace.get_path("ai_settings.yaml")
    ai_settings_file.write_text("")

    ai_config = AIConfig.load(str(ai_settings_file))
    assert ai_config.ai_name == ""
    assert ai_config.ai_role == ""
    assert ai_config.ai_goals == []
    assert ai_config.api_budget == 0.0
    assert ai_config.prompt_generator is None
    assert ai_config.command_registry is None


def test_ai_config_command_line_overrides():
    """
    Test command line overrides for AI parameters are set correctly in the AI config.
    """
    ai_config = AIConfig.load(
        ai_name="testGPT", ai_role="testRole", ai_goals=["testGoal"]
    )

    assert ai_config.ai_name == "testGPT"
    assert ai_config.ai_role == "testRole"
    assert ai_config.ai_goals == ["testGoal"]


def test_ai_config_command_line_overrides_with_config_file():
    """
    Test command line overrides for AI parameters are set correctly in the AI config.
    """
    ai_config = AIConfig.load(
        ai_name="testGPTOverride",
        ai_role="testRoleOverride",
        ai_goals=["testGoalOverride"],
        config_file="tests/unit/data/test_ai_config.yaml",
    )

    # Should have loaded from overrides and not from config
    assert ai_config.ai_name == "testGPTOverride"
    assert ai_config.ai_role == "testRoleOverride"
    assert ai_config.ai_goals == ["testGoalOverride"]


def test_ai_config_command_line_override_singular():
    """
    Test we can supply one override and prompt for the rest
    """

    ai_config = AIConfig.load(ai_name="testGPTOverride")

    assert ai_config.ai_name == "testGPTOverride"
    assert ai_config.ai_role == None
    assert ai_config.ai_goals == []
