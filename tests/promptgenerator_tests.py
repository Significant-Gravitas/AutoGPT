# Import the required libraries for unit testing
import os
import sys
import unittest

from autogpt.promptgenerator import PromptGenerator


# Create a test class for the PromptGenerator, subclassed from unittest.TestCase
class promptgenerator_tests(unittest.TestCase):
    # Set up the initial state for each test method by creating an instance of PromptGenerator
    def setUp(self):
        self.generator = PromptGenerator()

    # Test whether the add_constraint() method adds a constraint to the generator's constraints list
    def test_add_constraint(self):
        constraint = "Constraint1"
        self.generator.add_constraint(constraint)
        self.assertIn(constraint, self.generator.constraints)

    # Test whether the add_command() method adds a command to the generator's commands list
    def test_add_command(self):
        command_label = "Command Label"
        command_name = "command_name"
        args = {"arg1": "value1", "arg2": "value2"}
        self.generator.add_command(command_label, command_name, args)
        command = {
            "label": command_label,
            "name": command_name,
            "args": args,
        }
        self.assertIn(command, self.generator.commands)

    # Test whether the add_resource() method adds a resource to the generator's resources list
    def test_add_resource(self):
        resource = "Resource1"
        self.generator.add_resource(resource)
        self.assertIn(resource, self.generator.resources)

    # Test whether the add_performance_evaluation() method adds an evaluation to the generator's performance_evaluation list
    def test_add_performance_evaluation(self):
        evaluation = "Evaluation1"
        self.generator.add_performance_evaluation(evaluation)
        self.assertIn(evaluation, self.generator.performance_evaluation)

    # Test whether the generate_prompt_string() method generates a prompt string with all the added constraints, commands, resources and evaluations
    def test_generate_prompt_string(self):
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

        # Add all the constraints, commands, resources, and evaluations to the generator
        for constraint in constraints:
            self.generator.add_constraint(constraint)
        for command in commands:
            self.generator.add_command(
                command["label"], command["name"], command["args"]
            )
        for resource in resources:
            self.generator.add_resource(resource)
        for evaluation in evaluations:
            self.generator.add_performance_evaluation(evaluation)

        # Generate the prompt string and verify its correctness
        prompt_string = self.generator.generate_prompt_string()
        self.assertIsNotNone(prompt_string)
        for constraint in constraints:
            self.assertIn(constraint, prompt_string)
        for command in commands:
            self.assertIn(command["name"], prompt_string)

        # Check for each key-value pair in the command args dictionary
        for key, value in command["args"].items():
            self.assertIn(f'"{key}": "{value}"', prompt_string)
        for resource in resources:
            self.assertIn(resource, prompt_string)
        for evaluation in evaluations:
            self.assertIn(evaluation, prompt_string)
        self.assertIn("constraints", prompt_string.lower())
        self.assertIn("commands", prompt_string.lower())
        self.assertIn("resources", prompt_string.lower())
        self.assertIn("performance evaluation", prompt_string.lower())


# Run the tests when this script is executed
if __name__ == "__main__":
    unittest.main()
