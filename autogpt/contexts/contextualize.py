import json
import os

from autogpt.workspace import CONTEXTS_PATH

class ContextManager:
    _instance = None

    context_count = 0
    context_eval_count = 0

    def __new__(cls, context_directory=CONTEXTS_PATH, context_template_file=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.context_directory = context_directory
            cls._instance.context_template_file = context_template_file
            cls._instance.context_data = {}  # Add this line to initialize context_data
            cls._instance.read_context_template()
        return cls._instance


    def read_context_template(self):
        """
        Read the context template from the file specified during initialization.
        """
        with open(self.context_template_file, "r") as file:
            self.context_template = file.read()


    def is_valid_context(self, context_data, context_template):
        required_headers = [
            "# ",  # Context Name
            "## ",  # Context Goal
            "### Success Parameters",  # Success Parameters
            "### Guidance"  # Guidance
        ]

        for header in required_headers:
            if header not in context_data:
                return False

        # Additional checks can be added here, such as:
        # - Ensuring there are at least two success parameters
        # - Confirming that the context goal and success parameters are neither too specific nor too general
        # - Validating that the guidance section is not empty

        return True


    def create_new_context(self, context_name, context_data):
        """
        Create a new context using the provided name and data.

        :param context_name: Name of the new context
        :param context_data: Data to be included in the new context
        """
        if not self.is_valid_context(context_data, self.context_template):
            return "Invalid context data provided. Ensure you follow the template."

        self.context_count += 1
        self.context_eval_count = 0
        context_filename = f"{self.context_count}C - {context_name}.md"
        context_path = self.context_directory / context_filename

        with open(context_path, "w") as file:
            file.write(context_data)

        return f"New context with filename '{context_filename}' created successfully."
    

    def evaluate_context_success(self, context_name, summary_of_context_evaluation):
        """
        Save the evaluation summary of the context.

        :param context_name: Name of the context
        :param summary_of_context_evaluation: Summary of the context evaluation
        """
        # Increment the evaluation count for the current context
        self.context_eval_count += 1

        evaluation_filename = f"{self.context_count}C - {self.context_eval_count}E - {context_name}.md"
        evaluation_path = self.context_directory / evaluation_filename

        with open(evaluation_path, "w") as file:
            file.write(summary_of_context_evaluation)

        return f"Evaluation summary for context '{context_name}' saved successfully."
    

    def close_context(self, context_name, markdown_context_summary):
        """
        Close the context and save the summary.

        :param context_name: Name of the context
        :param markdown_context_summary: Markdown summary of the context
        :param success_status: Success status of the context
        """
        summary_filename = f"{self.context_count}C - Summary - {context_name}.md"
        summary_path = self.context_directory / summary_filename

        with open(summary_path, "w") as file:
            file.write(markdown_context_summary)

        return f"Context '{context_name}' closed successfully. Summary saved."


    def save_context_to_file(self):
        """
        Save the context data to a file.
        """
        with open(self.context_file, "w") as outfile:
            json.dump(self.context_data, outfile)


    def load_context_from_file(self, context_file):
        """
        Load context data from a file.

        :param context_file: Path to the context file
        """
        if os.path.exists(context_file):
            with open(context_file, "r") as file:
                self.context_data = json.load(file)


    def update_context(self, key, value):
        """
        Update a context with a new key and value.

        :param key: Key to be updated in the context
        :param value: New value for the key
        """
        self.context_data[key] = value


    def get_context(self):
        """
        Retrieve the context data.

        :return: Context data
        """
        return self.context_data
