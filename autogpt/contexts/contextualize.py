import json
import os

class ContextManager:
    _instance = None

    def __new__(cls, context_directory=None, context_template_file=None):
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


    def create_new_context(self, context_name, context_data):
        """
        Create a new context using the provided name and data.

        :param context_name: Name of the new context
        :param context_data: Data to be included in the new context
        """
        with open(f"{self.context_directory}/{context_name}.md", "w") as outfile:
            outfile.write(context_data.strip())


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
