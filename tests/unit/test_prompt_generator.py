from unittest import TestCase

from autogpt.prompts.generator import PromptGenerator


class TestPromptGenerator(TestCase):
    """
    Test cases for the PromptGenerator class, which is responsible for generating
    prompts for the AI with constraints, resources, and performance evaluations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the initial state for each test method by creating an instance of PromptGenerator.
        """
        cls.generator = PromptGenerator()

    # Test whether the add_constraint() method adds a constraint to the generator's constraints list
    def test_add_constraint(self):
        """
        Test if the add_constraint() method adds a constraint to the generator's constraints list.
        """
        constraint = "Constraint1"
        self.generator.add_constraint(constraint)
        self.assertIn(constraint, self.generator.constraints)

    def test_add_resource(self):
        """
        Test if the add_resource() method adds a resource to the generator's resources list.
        """
        resource = "Resource1"
        self.generator.add_resource(resource)
        self.assertIn(resource, self.generator.resources)

    def test_add_performance_evaluation(self):
        """
        Test if the add_performance_evaluation() method adds an evaluation to the generator's
        performance_evaluation list.
        """
        evaluation = "Evaluation1"
        self.generator.add_performance_evaluation(evaluation)
        self.assertIn(evaluation, self.generator.performance_evaluation)

    def test_generate_prompt_string(self):
        """
        Test if the generate_prompt_string() method generates a prompt string with all the added
        constraints, resources, and evaluations.
        """
        # Define the test data
        constraints = ["Constraint1", "Constraint2"]
        resources = ["Resource1", "Resource2"]
        evaluations = ["Evaluation1", "Evaluation2"]

        # Add test data to the generator
        for constraint in constraints:
            self.generator.add_constraint(constraint)
        for resource in resources:
            self.generator.add_resource(resource)
        for evaluation in evaluations:
            self.generator.add_performance_evaluation(evaluation)

        # Generate the prompt string and verify its correctness
        prompt_string = self.generator.generate_prompt_string()
        self.assertIsNotNone(prompt_string)

        # Check if all constraints, resources, and evaluations are present in the prompt string
        for constraint in constraints:
            self.assertIn(constraint, prompt_string)
        for resource in resources:
            self.assertIn(resource, prompt_string)
        for evaluation in evaluations:
            self.assertIn(evaluation, prompt_string)

        self.assertIn("constraints", prompt_string.lower())
        self.assertIn("resources", prompt_string.lower())
        self.assertIn("performance evaluation", prompt_string.lower())
