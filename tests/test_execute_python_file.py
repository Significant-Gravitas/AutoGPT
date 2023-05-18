import os
from io import StringIO
from unittest.mock import patch
import pytest
from autogpt.commands.execute_code import execute_python_file

@pytest.fixture
def temp_python_file(request):
    # Create a temporary Python file for testing
    filename = "test_file.py"
    with open(filename, "w") as f:
        f.write(request.param)
    yield filename
    # Clean up the temporary file
    os.remove(filename)

@pytest.mark.parametrize(
    "temp_python_file, args, expected_output",
    [
        pytest.param("import sys\nprint('Hello,', sys.argv[1])", "world!", "Hello, world!\n", id="valid_file_with_arguments"),
        pytest.param("print('Hello, world!')", "", "Hello, world!\n", id="valid_file_without_arguments"),
        pytest.param("", "", "Error: Invalid file type. Only .py files are allowed.", id="invalid_file"),
        pytest.param("print('Hello, world!')", "", "Error: File 'nonexistent_file.py' does not exist.", id="file_not_exist")
    ],
    indirect=["temp_python_file"]
)
def test_execute_python_file(temp_python_file, args, expected_output):
    # Patch the subprocess.run method to capture the output
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = expected_output

        # Execute the function with arguments
        output = execute_python_file(temp_python_file, args)

        # Assert the output matches the expected output
        assert output == expected_output