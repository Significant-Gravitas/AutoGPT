# File: test_prompt_generator.py
import pytest

from autogpt.prompts.generator import PromptGenerator


@pytest.fixture
def generator():
    return PromptGenerator()


def test_add_constraint(generator):
    constraint = "Constraint1"
    generator.add_constraint(constraint)
    assert constraint in generator.constraints

    constraints = ["Constraint1", "Constraint2", "Constraint3"]
    for constraint in constraints:
        generator.add_constraint(constraint)
    assert generator.constraints == constraints


def test_add_command(generator):
    command_label = "Command Label"
    command_name = "command_name"
    args = {"arg1": "value1", "arg2": "value2"}
    generator.add_command(command_label, command_name, args)
    command = {
        "label": command_label,
        "name": command_name,
        "args": args,
        "function": None,
    }
    assert command in generator.commands

    commands = [
        {
            "label": "Command1",
            "name": "command_name1",
            "args": {"arg1": "value1"},
        },
        {
            "label": "Command2",
            "name": "command_name2",
            "args": {},
        },
        {
            "label": "Command3",
            "name": "command_name3",
            "args": {"arg1": "value1", "arg2": "value2"},
        },
    ]
    for command in commands:
        generator.add_command(
            command["label"], command["name"], command["args"]
        )
    assert generator.commands == commands


def test_add_resource(generator):
    resource = "Resource1"
    generator.add_resource(resource)
    assert resource in generator.resources

    resources = ["Resource1", "Resource2", "Resource3"]
    for resource in resources:
        generator.add_resource(resource)
    assert generator.resources == resources


def test_add_performance_evaluation(generator):
    evaluation = "Evaluation1"
    generator.add_performance_evaluation(evaluation)
    assert evaluation in generator.performance_evaluation

    evaluations = ["Evaluation1", "Evaluation2", "Evaluation3"]
    for evaluation in evaluations:
        generator.add_performance_evaluation(evaluation)
    assert generator.performance_evaluation == evaluations


def test_generate_prompt_string(generator):
    prompt_string = generator.generate_prompt_string()
    assert prompt_string is not None
    assert prompt_string.strip() == ""

    constraints = ["Constraint1", "Constraint2"]
    commands = [
        {
            "label": "Command1",
            "name": "command_name1",
            "args": {"arg1": "value1"},
        },
        {
            "label": "Command2",
            "name": "command_name2",
            "args": {},
        },
    ]
    resources = ["Resource1", "Resource2"]
    evaluations = ["Evaluation1", "Evaluation2"]

    for constraint in constraints:
        generator.add_constraint(constraint)
    for command in commands:
        generator.add_command(
            command["label"], command["name"], command["args"]
        )
    for resource in resources:
        generator.add_resource(resource)
    for evaluation in evaluations:
        generator.add_performance_evaluation(evaluation)

    prompt_string = generator.generate_prompt_string()
    assert prompt_string is not None

    for constraint in constraints:
        assert constraint in prompt_string
    for command in commands:
        assert command["name"] in prompt_string
        for key, value in command["args"].items():
                        assert f'"{key}": "{value}"' in prompt_string
    for resource in resources:
        assert resource in prompt_string
    for evaluation in evaluations:
        assert evaluation in prompt_string

    assert "constraints" in prompt_string.lower()
    assert "commands" in prompt_string.lower()
    assert "resources" in prompt_string.lower()
    assert "performance evaluation" in prompt_string.lower()


def test_generate_prompt_string_with_repeated_items(generator):
    constraint = "Constraint1"
    command_label = "Command Label"
    command_name = "command_name"
    args = {"arg1": "value1", "arg2": "value2"}
    resource = "Resource1"
    evaluation = "Evaluation1"

    generator.add_constraint(constraint)
    generator.add_constraint(constraint)
    generator.add_command(command_label, command_name, args)
    generator.add_command(command_label, command_name, args)
    generator.add_resource(resource)
    generator.add_resource(resource)
    generator.add_performance_evaluation(evaluation)
    generator.add_performance_evaluation(evaluation)

    prompt_string = generator.generate_prompt_string()
    assert prompt_string is not None

    assert prompt_string.count(constraint) == 2
    assert prompt_string.count(command_name) == 2
    for key, value in args.items():
        assert prompt_string.count(f'"{key}": "{value}"') == 2
    assert prompt_string.count(resource) == 2
    assert prompt_string.count(evaluation) == 2

    assert "constraints" in prompt_string.lower()
    assert "commands" in prompt_string.lower()
    assert "resources" in prompt_string.lower()
    assert "performance evaluation" in prompt_string.lower()


