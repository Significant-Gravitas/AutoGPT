from io import StringIO
import os
import sys
import traceback
from RestrictedPython import compile_restricted, safe_globals


def execute_python_file(file):
    workspace_folder = "auto_gpt_workspace"

    if not file.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."
    
    try:
        # Prepend the workspace folder to the provided file name
        file_path = os.path.join(workspace_folder, file)

        # Check if the file exists
        if not os.path.isfile(file_path):
            return f"Error: File '{file}' does not exist."

        # Read the content of the file
        with open(file_path, 'r') as f:
            code = f.read()

        # Capture stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # Compile and execute the code in a restricted environment
        try:
            restricted_code = compile_restricted(code, '<inline>', 'exec')
            exec(restricted_code, safe_globals)
        except Exception as e:
            result = f"Error while executing code:\n{traceback.format_exc()}"
        else:
            result = sys.stdout.getvalue()
            
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        return result
    except Exception as e:
        return f"Error: {str(e)}"