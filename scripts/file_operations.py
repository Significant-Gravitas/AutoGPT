import os
import os.path
from auto_gpt.commands import command


# Set a dedicated folder for file I/O
working_directory = "auto_gpt_workspace"

if not os.path.exists(working_directory):
    os.makedirs(working_directory)


def safe_join(base, *paths):
    new_path = os.path.join(base, *paths)
    norm_new_path = os.path.normpath(new_path)

    if os.path.commonprefix([base, norm_new_path]) != base:
        raise ValueError("Attempted to access outside of working directory.")

    return norm_new_path

@command("read_file", "Read file", '"file": "<file>"')
def read_file(file):
    try:
        filepath = safe_join(working_directory, file)
        with open(filepath, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return "Error: " + str(e)

@command("write_to_file", "Write to file", '"file": "<file>", "text": "<text>"')
def write_to_file(file, text):
    try:
        filepath = safe_join(working_directory, file)
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, "w") as f:
            f.write(text)
        return "File written to successfully."
    except Exception as e:
        return "Error: " + str(e)

@command("append_to_file", "Append to file", '"file": "<file>", "text": "<text>"')
def append_to_file(file, text):
    try:
        filepath = safe_join(working_directory, file)
        with open(filepath, "a") as f:
            f.write(text)
        return "Text appended successfully."
    except Exception as e:
        return "Error: " + str(e)

@command("delete_file", "Delete file", '"file": "<file>"')
def delete_file(file):
    try:
        filepath = safe_join(working_directory, file)
        os.remove(filepath)
        return "File deleted successfully."
    except Exception as e:
        return "Error: " + str(e)

@command("search_files", "Search Files", '"directory": "<directory>"')
def search_files(directory):
    found_files = []

    if directory == "" or directory == "/":
        search_directory = working_directory
    else:
        search_directory = safe_join(working_directory, directory)

    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.startswith('.'):
                continue
            relative_path = os.path.relpath(os.path.join(root, file), working_directory)
            found_files.append(relative_path)

    return found_files