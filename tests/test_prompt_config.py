from autogpt.config.prompt_config import PromptConfig

"""
Test cases for the PromptConfig class, which handles loads the Prompts configuration
settings from a YAML file.
"""


def test_prompt_config_loading(tmp_path):
    """Test if the prompt configuration loads correctly"""

    yaml_content = """
constraints:
- A test constraint
- Another test constraint
- A third test constraint
resources:
- A test resource
- Another test resource
- A third test resource
performance_evaluations:
- A test performance evaluation
- Another test performance evaluation
- A third test performance evaluation
"""
    config_file = tmp_path / "test_prompt_settings.yaml"
    config_file.write_text(yaml_content)

    prompt_config = PromptConfig(config_file)

    assert len(prompt_config.constraints) == 3
    assert prompt_config.constraints[0] == "A test constraint"
    assert prompt_config.constraints[1] == "Another test constraint"
    assert prompt_config.constraints[2] == "A third test constraint"
    assert len(prompt_config.resources) == 3
    assert prompt_config.resources[0] == "A test resource"
    assert prompt_config.resources[1] == "Another test resource"
    assert prompt_config.resources[2] == "A third test resource"
    assert len(prompt_config.performance_evaluations) == 3
    assert prompt_config.performance_evaluations[0] == "A test performance evaluation"
    assert (
        prompt_config.performance_evaluations[1]
        == "Another test performance evaluation"
    )
    assert (
        prompt_config.performance_evaluations[2]
        == "A third test performance evaluation"
    )
