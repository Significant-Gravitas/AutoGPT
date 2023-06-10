import yaml

from autogpt.config.ai_config import AIConfig

"""
Test cases for the AIConfig class, which handles loads the AI configuration
settings from a YAML file.
"""


def test_goals_are_always_lists_of_strings(tmp_path):
    """Test if the goals attribute is always a list of strings."""

    yaml_content = """configs:
      McFamished:
        ai_goals:
        - 'Goal 1: Make a sandwich'
        - 'Goal 2, Eat the sandwich'
        - 'Goal 3 - Go to sleep'
        - 'Goal 4: Wake up'
        ai_role: A hungry AI
        api_budget: 0.0
    """

    config_file = tmp_path / "ai_settings.yaml"
    config_file.write_text(yaml_content)

    ai_config = AIConfig.load("McFamished", config_file)

    assert len(ai_config.ai_goals) == 4
    assert ai_config.ai_goals[0] == "Goal 1: Make a sandwich"
    assert ai_config.ai_goals[1] == "Goal 2, Eat the sandwich"
    assert ai_config.ai_goals[2] == "Goal 3 - Go to sleep"
    assert ai_config.ai_goals[3] == "Goal 4: Wake up"

    config_file.write_text("")
    ai_config.save(config_file)

    saved_yaml = yaml.safe_load(config_file.read_text())

    expected_yaml = yaml.safe_load(
        """configs:
      McFamished:
        ai_goals:
        - 'Goal 1: Make a sandwich'
        - 'Goal 2, Eat the sandwich'
        - 'Goal 3 - Go to sleep'
        - 'Goal 4: Wake up'
        ai_role: A hungry AI
        api_budget: 0.0
        plugins: []
    """
    )

    assert saved_yaml == expected_yaml


def test_ai_config_file_not_exists(workspace):
    """Test if file does not exist."""

    config_file = workspace.get_path("ai_settings.yaml")

    ai_config = AIConfig.load("Test", str(config_file))
    assert ai_config is None


def test_ai_config_file_is_empty(workspace):
    """Test if file does not exist."""

    config_file = workspace.get_path("ai_settings.yaml")
    config_file.write_text("")

    ai_config = AIConfig.load("Test", str(config_file))
    assert ai_config is None


def test_delete_method(tmp_path):
    """Test if the delete method properly removes an AI configuration from the file."""

    yaml_content = """configs:
      AI1:
        ai_goals:
        - Goal 1
        ai_role: Test role
        api_budget: 0.0
      AI2:
        ai_goals:
        - Goal 2
        ai_role: Another role
        api_budget: 0.0
    """

    config_file = tmp_path / "ai_settings.yaml"
    config_file.write_text(yaml_content)

    AIConfig().delete(
        config_file,
        "AI1",
    )

    # Print file contents after deletion
    print(config_file.read_text())

    ai_config = AIConfig.load("AI1", str(config_file))
    assert ai_config is None

    ai_config2 = AIConfig.load("AI2", str(config_file))
    assert ai_config2 is not None

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None
    ai_config2 = None


def test_special_character_config(tmp_path):
    yaml_content = """configs:
      SpécialAI:
        ai_goals:
        - 'Gôal 1: Mäke a sàndwich'
        ai_role: 'A hùngry AI'
        api_budget: 0.0
    """

    config_file = tmp_path / "ai_settings.yaml"
    config_file.write_text(yaml_content, encoding='utf-8')

    ai_config = AIConfig.load("SpécialAI", config_file)

    assert ai_config.ai_goals == ["Gôal 1: Mäke a sàndwich"]
    assert ai_config.ai_role == "A hùngry AI"
    assert ai_config.api_budget == 0.0

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None


def test_handling_special_characters_configuration(tmp_path):
    config_file = tmp_path / "ai_settings.yaml"
    config_file.write_text(
        "configs:\n  AI1:\n    ai_goals: ['Goal with special characters: !@#$%^&*()']\n"
    )

    ai_config = AIConfig.load("AI1", config_file)

    assert len(ai_config.ai_goals) == 1
    assert ai_config.ai_goals[0] == "Goal with special characters: !@#$%^&*()"

    ai_config.save(config_file)

    saved_yaml = yaml.safe_load(config_file.read_text())
    expected_yaml = {
        "configs": {
            "AI1": {
                "ai_goals": ["Goal with special characters: !@#$%^&*()"],
            }
        }
    }
    assert (
        saved_yaml["configs"]["AI1"]["ai_goals"][0]
        == "Goal with special characters: !@#$%^&*()"
    )

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None


def test_loading_large_configuration(tmp_path):
    config_file = tmp_path / "ai_settings.yaml"

    # Create a large configuration with 100 AI entries
    config_content = "configs:\n"
    for i in range(100):
        config_content += f"  AI{i+1}:\n    ai_goals: ['Goal {i+1}']\n    ai_role: test role {i+1}\n    api_budget: 0.0\n    plugins: []\n"

    config_file.write_text(config_content)

    ai_config = AIConfig.load("AI50", config_file)

    print(config_content)  # debug: print the generated content

    assert ai_config.ai_name == "AI50"
    assert ai_config.ai_goals == ["Goal 50"]
    assert ai_config.api_budget == 0.0

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None
    config_content = None


def test_saving_large_configuration(tmp_path):
    config_file = tmp_path / "ai_settings.yaml"
    ai_config = AIConfig("AI1", ai_goals=["Goal 1"])

    # Create a large configuration with 100 AI entries
    config_content = "configs:\n"
    for i in range(100):
        config_content += f"  AI{i+1}:\n    ai_goals: ['Goal {i+1}']\n    ai_role: test role {i+1}\n    api_budget: 0.0\n    plugins: []\n"

    config_file.write_text(config_content)

    ai_config.save(config_file)

    saved_yaml = yaml.safe_load(config_file.read_text())
    expected_yaml = {
        "configs": {"AI1": {"ai_goals": ["Goal 1"], "ai_role": "", "api_budget": 0.0, "plugins": []}}
    }

    assert saved_yaml == expected_yaml

    # Clean up the configuration file and related variables
    config_file.unlink()
    ai_config = None
    config_content = None
