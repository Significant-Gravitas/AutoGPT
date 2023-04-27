import pathlib
import random
import string
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import autogpt.commands.execute_code as sut


class TestExecuteCode(TestCase):
    def test_execute_python_file(self):
        random_string = "".join(
            random.choice(string.ascii_lowercase) for _ in range(10)
        )
        with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as temp_file:
            temp_file.write(str.encode(f"print('Hello {random_string}!')"))
            temp_file.flush()
            temp_file_path = pathlib.Path(temp_file.name).parent
            config_mock = MagicMock(wraps=sut.CFG, workspace_path=temp_file_path)
            with patch("autogpt.commands.execute_code.CFG", config_mock):
                result = sut.execute_python_file(temp_file.name)
                self.assertEqual(result, f"Hello {random_string}!\n")
