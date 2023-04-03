from pathlib import Path

# Set a dedicated folder for file I/O
working_directory = Path("auto_gpt_workspace")
working_directory.mkdir(exist_ok=True)

def safe_join(base, *paths):
    new_path = base.joinpath(*paths)
    norm_new_path = new_path.resolve()

    if base not in norm_new_path.parents:
        raise ValueError("Attempted to access outside of working directory.")

    return norm_new_path

def read_file(filename):
    try:
        filepath = safe_join(working_directory, filename)
        content = filepath.read_text()
        return content
    except Exception as e:
        return "Error: " + str(e)

def write_to_file(filename, text):
    try:
        filepath = safe_join(working_directory, filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(text)
        return "File written to successfully."
    except Exception as e:
        return "Error: " + str(e)

def append_to_file(filename, text):
    try:
        filepath = safe_join(working_directory, filename)
        with filepath.open("a") as f:
            f.write(text)
        return "Text appended successfully."
    except Exception as e:
        return "Error: " + str(e)

def delete_file(filename):
    try:
        filepath = safe_join(working_directory, filename)
        filepath.unlink()
        return "File deleted successfully."
    except Exception as e:
        return "Error: " + str(e)
