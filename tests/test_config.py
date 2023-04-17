from unittest import TestCase
from unittest.mock import patch, call
import os
import tempfile
from _pytest import pathlib
from autogpt.config import Config


class TestConfig(TestCase):
    """
    Test cases for the Config class, which handles the configuration settings
    for the AI and ensures it behaves as a singleton.
    """
    @patch('config.load_dotenv', autospec=True)
    @patch('config.dotenv_values', autospec=True)
    @patch('builtins.input', autospec=True, side_effect=['y', '8192'])
    @patch('config.set_key', autospec=True)
    @patch('config.unset_key', autospec=True)
    @patch('config.Path', autospec=True)

    def setUp(self):
        """
        Set up the test environment by creating an instance of the Config class.
        """
        self.config = Config()
     
    def setUp(self, mock_path, mock_unset_key, mock_set_key, mock_input,
              mock_dotenv_values, mock_load_dotenv):
        self.config = Config()

    def test_update_env_from_template(self):
        # create temporary files for testing
        env_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        template_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        env_file.close()
        template_file.close()

        # patch Path object to return temporary file paths
        with patch("pathlib.Path", wraps=pathlib.Path) as mock_path:
            mock_env_file = mock_path.return_value
            mock_template_file = mock_path.return_value
            mock_env_file.is_file.return_value = True
            mock_template_file.is_file.return_value = True
            mock_env_file.resolve.return_value = mock_env_file
            mock_template_file.resolve.return_value = mock_template_file
            mock_env_file.__truediv__.return_value = env_file.name
            mock_template_file.__truediv__.return_value = template_file.name

            # patch dotenv functions
            with patch("dotenv.load_dotenv", spec=True) as mock_load_dotenv, \
                    patch("dotenv.dotenv_values", spec=True) as mock_dotenv_values, \
                    patch("builtins.input", side_effect=["y", "8192"]):
                # set up mock return values for dotenv_values
                mock_env_values = {"ENV_VAR": "env_value"}
                mock_template_values = {"ENV_VAR": "template_value", "NEW_VAR": ""}
                mock_dotenv_values.side_effect = [mock_env_values, mock_template_values]

                # call the method
                config = Config()
                config.update_env_from_template()

                # assert that load_dotenv was called with the template file path
                mock_load_dotenv.assert_called_once_with(mock_template_file, verbose=True)

                # assert that dotenv_values was called twice, once for each file
                calls = [call(mock_env_file), call(mock_template_file)]
                mock_dotenv_values.assert_has_calls(calls)

                # assert that the expected changes were made to the .env file
                with open(env_file.name, mode="r") as f:
                    content = f.read()
                expected_content = "ENV_VAR=env_value\nNEW_VAR=\n"
                self.assertEqual(content, expected_content)

                # set the value of the NEW_VAR key in the .env file
                config.set_key("NEW_VAR", "new_value")

                # assert that the value of the NEW_VAR key in the .env file was updated
                with open(env_file.name, mode="r") as f:
                    content = f.read()
                expected_content = "ENV_VAR=env_value\nNEW_VAR=new_value\n"
                self.assertEqual(content, expected_content)

        # remove temporary files
        os.unlink(env_file.name)
        os.unlink(template_file.name)

    def test_singleton(self):
        """
        Test if the Config class behaves as a singleton by ensuring that two instances are the same.
        """
        config2 = Config()
        self.assertIs(self.config, config2)

    def test_initial_values(self):
        """
        Test if the initial values of the Config class attributes are set correctly.
        """
        self.assertFalse(self.config.debug_mode)
        self.assertFalse(self.config.continuous_mode)
        self.assertFalse(self.config.speak_mode)
        self.assertEqual(self.config.fast_llm_model, "gpt-3.5-turbo")
        self.assertEqual(self.config.smart_llm_model, "gpt-4")
        self.assertEqual(self.config.fast_token_limit, 4000)
        self.assertEqual(self.config.smart_token_limit, 8000)

    def test_set_continuous_mode(self):
        """
        Test if the set_continuous_mode() method updates the continuous_mode attribute.
        """
        self.config.set_continuous_mode(True)
        self.assertTrue(self.config.continuous_mode)

    def test_set_speak_mode(self):
        """
        Test if the set_speak_mode() method updates the speak_mode attribute.
        """
        self.config.set_speak_mode(True)
        self.assertTrue(self.config.speak_mode)

    def test_set_fast_llm_model(self):
        """
        Test if the set_fast_llm_model() method updates the fast_llm_model attribute.
        """
        self.config.set_fast_llm_model("gpt-3.5-turbo-test")
        self.assertEqual(self.config.fast_llm_model, "gpt-3.5-turbo-test")

    def test_set_smart_llm_model(self):
        """
        Test if the set_smart_llm_model() method updates the smart_llm_model attribute.
        """
        self.config.set_smart_llm_model("gpt-4-test")
        self.assertEqual(self.config.smart_llm_model, "gpt-4-test")

    def test_set_fast_token_limit(self):
        """
        Test if the set_fast_token_limit() method updates the fast_token_limit attribute.
        """
        self.config.set_fast_token_limit(5000)
        self.assertEqual(self.config.fast_token_limit, 5000)

    def test_set_smart_token_limit(self):
        """
        Test if the set_smart_token_limit() method updates the smart_token_limit attribute.
        """
        self.config.set_smart_token_limit(9000)
        self.assertEqual(self.config.smart_token_limit, 9000)

    def test_set_debug_mode(self):
        """
        Test if the set_debug_mode() method updates the debug_mode attribute.
        """
        self.config.set_debug_mode(True)
        self.assertTrue(self.config.debug_mode)
