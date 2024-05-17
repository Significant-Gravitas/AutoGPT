from forge.config.ai_profile import AIProfile
from forge.file_storage import FileStorage

"""
Test cases for the AIProfile class, which handles loads the AI configuration
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

    ai_profile = AIProfile.load(ai_settings_file)

    assert len(ai_profile.ai_goals) == 4
    assert ai_profile.ai_goals[0] == "Goal 1: Make a sandwich"
    assert ai_profile.ai_goals[1] == "Goal 2, Eat the sandwich"
    assert ai_profile.ai_goals[2] == "Goal 3 - Go to sleep"
    assert ai_profile.ai_goals[3] == "Goal 4: Wake up"

    ai_settings_file.write_text("")
    ai_profile.save(ai_settings_file)

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


def test_ai_profile_file_not_exists(storage: FileStorage):
    """Test if file does not exist."""

    ai_settings_file = storage.get_path("ai_settings.yaml")

    ai_profile = AIProfile.load(str(ai_settings_file))
    assert ai_profile.ai_name == ""
    assert ai_profile.ai_role == ""
    assert ai_profile.ai_goals == []
    assert ai_profile.api_budget == 0.0


def test_ai_profile_file_is_empty(storage: FileStorage):
    """Test if file does not exist."""

    ai_settings_file = storage.get_path("ai_settings.yaml")
    ai_settings_file.write_text("")

    ai_profile = AIProfile.load(str(ai_settings_file))
    assert ai_profile.ai_name == ""
    assert ai_profile.ai_role == ""
    assert ai_profile.ai_goals == []
    assert ai_profile.api_budget == 0.0
