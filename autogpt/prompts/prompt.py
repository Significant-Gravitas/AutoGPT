from autogpt.config.config import Config
from autogpt.config.prompt_config import PromptConfig
from autogpt.prompts.generator import PromptGenerator

DEFAULT_TRIGGERING_PROMPT = "Determine exactly one command to use, and respond using the JSON schema specified previously:"


def build_default_prompt_generator(config: Config) -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Initialize the PromptConfig object and load the file set in the main config (default: prompts_settings.yaml)
    prompt_config = PromptConfig(config.prompt_settings_file)

    # Add constraints to the PromptGenerator object
    for constraint in prompt_config.constraints:
        prompt_generator.add_constraint(constraint)

    # Add resources to the PromptGenerator object
    for resource in prompt_config.resources:
        prompt_generator.add_resource(resource)

    # Add performance evaluations to the PromptGenerator object
    for performance_evaluation in prompt_config.performance_evaluations:
        prompt_generator.add_performance_evaluation(performance_evaluation)

    return prompt_generator
