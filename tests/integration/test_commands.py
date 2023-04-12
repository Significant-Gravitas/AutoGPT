import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path
import unittest
import platform

# Import necessary modules for testing, file operations, and handling system paths.

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path / 'scripts'))
from file_operations import read_file
from file_operations import delete_file

env_vars = {
    'MEMORY_BACKEND': 'no_memory',
    'TEMPERATURE': "0"
}


class TestCommands(unittest.TestCase):

    def test_write_file(self):
        # Test case to check if the write_file command can successfully write 'Hello World' to a file
        # named 'hello_world.txt'.

        # Read the current ai_settings.yaml file and store its content.
        ai_settings = None
        with contextlib.suppress(Exception):
            with open('ai_settings.yaml', 'r') as f:
                ai_settings = f.read()

        try:
            with contextlib.suppress(Exception):
                # Clean up any existing 'hello_world.txt' file before testing.
                delete_file('hello_world.txt')

                # Remove ai_settings.yaml file to avoid continuing from the previous session.
                os.remove('ai_settings.yaml')

            # Prepare input data for the test.
            input_data = '''write_file-GPT
an AI designed to use the write_file command to write 'Hello World' into a file named "hello_world.txt" and then use the task_complete command to complete the task.
Use the write_file command to write 'Hello World' into a file named "hello_world.txt".
Use the task_complete command to complete the task.
Do not use any other commands.

y -5
EOF'''
            script_path = root_path / 'scripts' / 'main.py'
            command = f'{sys.executable} {script_path}'

            # Execute the script with the input data.
            process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True, env={**os.environ, **env_vars})
            process.communicate(input_data.encode())

            # Read the content of the 'hello_world.txt' file created during the test.
            content = read_file('hello_world.txt')
        finally:
            if ai_settings:
                # Restore the original ai_settings.yaml file.
                with open('ai_settings.yaml', 'w') as f:
                    f.write(ai_settings)

        # Check if the content of the 'hello_world.txt' file is equal to 'Hello World'.
        self.assertEqual(content, 'Hello World', f"Expected 'Hello World', got {content}")


# Run the test case.
if __name__ == '__main__':
    unittest.main()
