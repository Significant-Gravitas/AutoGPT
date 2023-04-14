import os

class FileManager:
    def __init__(self, working_directory=None):
        if working_directory is None:
            working_directory = os.path.join("", "auto_gpt_workspace")
        
        self.working_directory = working_directory

        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)

    def safe_join(self, base, *paths):
        new_path = os.path.join(base, *paths)
        norm_new_path = os.path.normpath(new_path)

        if os.path.commonprefix([base, norm_new_path]) != base:
            raise ValueError("Attempted to access outside of working directory.")

        return norm_new_path

    def read_file(self, filename):
        try:
            filepath = self.safe_join(self.working_directory, filename)
            with open(filepath, "r", encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error: {e}"

    def write_to_file(self, filename, text):
        try:
            filepath = self.safe_join(self.working_directory, filename)
            directory = os.path.dirname(filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(text)
            return "File written to successfully."
        except Exception as e:
            return f"Error: {e}"

    def append_to_file(self, filename, text):
        try:
            filepath = self.safe_join(self.working_directory, filename)
            with open(filepath, "a", encoding='utf-8') as f:
                f.write(text)
            return "Text appended successfully."
        except Exception as e:
            return f"Error: {e}"

    def delete_file(self, filename):
        try:
            filepath = self.safe_join(self.working_directory, filename)
            os.remove(filepath)
            return "File deleted successfully."
        except Exception as e:
            return f"Error: {e}"

    def search_files(self, directory=""):
        search_directory = self.working_directory if directory == "" or directory == "/" else self.safe_join(self.working_directory, directory)

        found_files = [os.path.relpath(os.path.join(root, file), self.working_directory)
                       for root, _, files in os.walk(search_directory)
                       for file in files if not file.startswith('.')]

        return found_files
