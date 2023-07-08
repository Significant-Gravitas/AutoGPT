from autogpt.prompts.generator import PromptGenerator


def test_add_constraint():
    """
    Test if the add_constraint() method adds a constraint to the generator's constraints list.
    """
    constraint = "Constraint1"
    generator = PromptGenerator()
    generator.add_constraint(constraint)
    assert constraint in generator.constraints


def test_add_command():
    """
    Test if the add_command() method adds a command to the generator's commands list.
    """
    command_label = "Command Label"
    command_name = "command_name"
    params = {"arg1": "value1", "arg2": "value2"}
    generator = PromptGenerator()
    generator.add_command(command_label, command_name, params)
    command = {
        "label": command_label,
        "name": command_name,
        "params": params,
        "function": None,
    }
    assert command in generator.commands


def test_add_resource():
    """
    Test if the add_resource() method adds a resource to the generator's resources list.
    """
    resource = "Resource1"
    generator = PromptGenerator()
    generator.add_resource(resource)
    assert resource in generator.resources


def test_add_performance_evaluation():
    """
    Test if the add_performance_evaluation() method adds an evaluation to the generator's
    performance_evaluation list.
    """
    evaluation = "Evaluation1"
    generator = PromptGenerator()
    generator.add_performance_evaluation(evaluation)
    assert evaluation in generator.performance_evaluation


def test_generate_prompt_string(config):
    """
    Test if the generate_prompt_string() method generates a prompt string with all the added
    constraints, commands, resources, and evaluations.
    """

    # Define the test data
    constraints = ["Constraint1", "Constraint2"]
    commands = [
        {
            "label": "Command1",
            "name": "command_name1",
            "params": {"arg1": "value1"},
        },
        {
            "label": "Command2",
            "name": "command_name2",
            "params": {},
        },
    ]
    resources = ["Resource1", "Resource2"]
    evaluations = ["Evaluation1", "Evaluation2"]

    # Add test data to the generator
    generator = PromptGenerator()
    for constraint in constraints:
        generator.add_constraint(constraint)
    for command in commands:
        generator.add_command(command["label"], command["name"], command["params"])
    for resource in resources:
        generator.add_resource(resource)
    for evaluation in evaluations:
        generator.add_performance_evaluation(evaluation)

    # Generate the prompt string and verify its correctness
    prompt_string = generator.generate_prompt_string(config)
    assert prompt_string is not None

    # Check if all constraints, commands, resources, and evaluations are present in the prompt string
    for constraint in constraints:
        assert constraint in prompt_string
    for command in commands:
        assert command["name"] in prompt_string
        for key, value in command["params"].items():
            assert f'"{key}": "{value}"' in prompt_string
    for resource in resources:
        assert resource in prompt_string
    for evaluation in evaluations:
        assert evaluation in prompt_string
