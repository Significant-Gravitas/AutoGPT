import json
import os
from pathlib import Path
import re
from autogpt.contexts.templates import TemplateManager

from autogpt.workspace import CONTEXTS_PATH
from autogpt.contexts.markdown_parsing import get_header_levels, matches_template

class ContextManager:
    _instance = None
    template_manager = TemplateManager()

    def get_context_manager_instance():
        return ContextManager()

    max_active_contexts = 2

    context_count = 0
    open_context_count = 0
    context_eval_count = 0
    context_template = None

    current_context = None
    current_context_folder = None
    current_context_path = None

    def __new__(cls, context_directory=CONTEXTS_PATH, context_template_file=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.context_directory = context_directory
            cls._instance.context_template_file = context_template_file
            cls._instance.context_data = {}
            cls._instance.current_context = None
            cls._instance.copy_template_to_directory()
            cls._instance.read_context_template()
            cls._instance.load_all_contexts()
        return cls._instance


    def copy_template_to_directory(self):
        """
        Copy the default context template to the context directory.
        """
        current_script_path = Path(__file__).resolve().parent
        template_path = current_script_path / "context_template.md"

        destination_file = self.context_directory / "context_template.md"
        if not destination_file.exists():
            with open(template_path, "r") as src_file, open(destination_file, "w") as dst_file:
                dst_file.write(src_file.read())


    # CONTEXT HELPERS

    def read_context_template(self):
        """
        Read the default context template.
        """
        with open(self.context_template_file, "r") as file:
            self.context_template = file.read()

        print("Using Context Template ------- \n")
        print(self.context_template)


    def matches_context_template(self, context_data, template):
        """
        Validate context_data by checking the pattern of markdown headers.

        :param context_data: Data to be included in the new context
        :param template: Template to validate the context_data against
        :return: Boolean indicating if the context_data is valid
        """
        # print(f"Context Data:\n{context_data}")
        # print(f"Template:\n{template}")

        if matches_template(context_data, template):
            return "New context created successfully."
        else:
            context_data_header_levels = get_header_levels(context_data)
            template_header_levels = get_header_levels(template)

            return f"This markdown does not match the template.\nTemplate Header Levels: {template_header_levels}\nContext Data Header Levels: {context_data_header_levels}\n\nYour current primary objective is to create a context from this template:\n {self.context_template}\n"


    # CONTEXT LIFECYCLE

    # Modified this method to update the 'current_context_folder' attribute
    def create_context_folder(self, context_name):
        """
        Create a folder for the specific context.

        :param context_name: Name of the context
        :return: Path of the created folder
        """
        context_folder = self.context_directory / context_name
        context_folder.mkdir(exist_ok=True)
        self.current_context_folder = context_folder
        return context_folder


    def create_new_context(self, context_name, context_data, template=None):
        """
        Create a new context using the provided name and data.

        :param context_name: Name of the new context
        :param context_data: Data to be included in the new context
        :param template: Optional template to use when creating the context
        """

        if (self.open_context_count >= self.max_active_contexts):
            return f"There are already {self.max_active_contexts} contexts open. Please conclude a context before creating a new one. Current context: {self.current_context}"
        
        if template is not None:
            if not self.matches_context_template(context_data, template):
                return self.matches_context_template(context_data, template)
        elif not self.matches_context_template(context_data, self.template_manager.get_default_template()):
            return self.matches_context_template(context_data, self.template_manager.get_default_template())

        self.context_count += 1
        self.open_context_count += 1
        self.context_eval_count = 0
        context_filename = f"C{self.context_count}-{context_name}.md"

        current_context_folder = self.create_context_folder(context_name)
        current_context_path = current_context_folder / context_filename

        with open(current_context_path, "w") as file:
            file.write(context_data)

        self.context_data[context_name] = current_context_path
        self.current_context = context_name

        return f"New context at path '{current_context_path}' created successfully."


    def evaluate_context_success(self, context_name, summary_of_context_evaluation):
        """
        Save the evaluation summary of the context.

        :param context_name: Name of the context
        :param summary_of_context_evaluation: Summary of the context evaluation
        """
        self.context_eval_count += 1

        evaluation_filename = f"C{self.context_count}-{self.context_eval_count}-Eval-{context_name}.md"
        evaluation_path = self.current_context_folder / evaluation_filename

        with open(evaluation_path, "w") as file:
            file.write(summary_of_context_evaluation)

        return f"Evaluation summary for context '{context_name}' saved successfully."
    

    def close_context(self, context_name, markdown_context_summary):
        """
        Close the context and save the summary.

        :param context_name: Name of the context
        :param markdown_context_summary: Markdown summary of the context
        """
        summary_filename = f"context-{self.context_count}-summary-{context_name}.md"
        summary_path = self.current_context_folder / summary_filename

        with open(summary_path, "w") as file:
            file.write(markdown_context_summary)

        self.open_context_count -= 1
        return f"Context '{context_name}' closed successfully. Summary saved."


    # CONTEXT MANAGEMENT

    def update_context(self, context_name, context_data):
        """
        Update a context with new context_data.

        :param context_name: Name of the context to be updated
        :param context_data: New context data in markdown format
        """
        if not self.matches_context_template(context_data):
            return f"This markdown does not match the template. Ensure you follow the template:\n {self.context_template}"

        self.context_data[context_name] = context_data


    def get_context(self, context_name):
        """
        Retrieve the context data of a specific context.

        :param context_name: Name of the context to retrieve
        :return: Context data or an error message if not found
        """
        if context_name in self.context_data:
            return self.context_data[context_name]
        else:
            return f"Context '{context_name}' not found. {self.list_contexts()}"

    
    def get_current_context(self):
        """
        Retrieve the current context data.

        :return: Current context data
        """
        if self.current_context:
            return self.context_data[self.current_context]
        else:
            return "No context is currently selected."
    
    def has_active_context(self):
        for context in self.contexts:
            if context["status"] == "active":
                return True
        return False

    def load_all_contexts(self):
        """
        Load all context files from the context directory.
        """
        self.context_data = {}
        for context_file in self.context_directory.glob("context-*.md"):
            context_name = context_file.stem
            with open(context_file, "r") as file:
                self.context_data[context_name] = file.read()


    def list_contexts(self):
        """
        List all available contexts.

        :return: List of context names
        """
        return f"All contexts: {list(self.context_data.keys())}"


    def switch_context(self, context_name):
        """
        Switch to a specific context.

        :param context_name: Name of the context to switch to
        :return: Status message
        """
        if context_name in self.context_data:
            self.current_context = context_name
            # Update the 'current_context_folder' attribute when switching contexts
            self.current_context_folder = self.context_directory / context_name
            print(f"Switched to context '{context_name}'.")
            print(f"Context file contents: {self.context_data[context_name]}")
            return f"Switched to context '{context_name}'."
        else:
            return f"Context '{context_name}' not found. {self.list_contexts()}"
        

    def merge_contexts(self, context_name_1, context_name_2, new_context_name, new_context_data):
        """
        Merge multiple contexts into a single context.

        :param context_names: List of context names to be merged
        :param new_context_name: Name of the new merged context
        :param new_context_data: Data to be included in the new merged context
        :return: Status message
        """
        if not self.matches_context_template(new_context_data, self.context_template):
            return f"This markdown does not match the template. Ensure you follow the template:\n {self.context_template}"

        for context_name in [context_name_1, context_name_2]:
            if context_name not in self.context_data:
                return f"Context '{context_name}' not found."

        return self.create_new_context(new_context_name, new_context_data)
    
    def get_context_statistics(self):
        """
        Get statistics on the available contexts.

        :return: A dictionary containing context statistics
        """
        # Customize this method to return statistics relevant to your use case
        statistics = {
            "total_contexts": len(self.context_data),
            "current_context": self.current_context,
        }
        return statistics

    def delete_context(self, context_name):
        """
        Delete a context and its associated files.

        :param context_name: Name of the context to delete
        :return: Status message
        """
        if context_name not in self.context_data:
            return f"Context '{context_name}' not found. {self.list_contexts()}"

        del self.context_data[context_name]

        context_file = self.context_directory / f"{context_name}.md"
        if context_file.exists():
            context_file.unlink()

        eval_files = list(self.context_directory.glob(f"{context_name}-eval-*.md"))
        for eval_file in eval_files:
            if eval_file.exists():
                eval_file.unlink()

        summary_file = self.context_directory / f"{context_name}-summary.md"
        if summary_file.exists():
            summary_file.unlink()

        return f"Context '{context_name}' and its associated files have been deleted."
