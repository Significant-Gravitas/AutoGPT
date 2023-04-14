import pathlib
import random
import string
import tempfile
from unittest import TestCase
from unittest.mock import patch

from autogpt.execute_code import execute_python_file


class TestExecuteCode(TestCase):

    def test_execute_python_file(self):
        random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        with tempfile.NamedTemporaryFile(delete=True, suffix='.py') as temp_file:
            temp_file.write(str.encode(f"print('Hello {random_string}!')"))
            temp_file.flush()
            temp_file_path = pathlib.Path(temp_file.name).parent
            temp_file_name = pathlib.Path(temp_file.name).name
            with patch("autogpt.execute_code.WORKSPACE_FOLDER", temp_file_path):
                result = execute_python_file(temp_file_name)
                self.assertEqual(result, f"Hello {random_string}!\n")
