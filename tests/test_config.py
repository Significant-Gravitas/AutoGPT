from unittest import TestCase

from autogpt.config import Config


class TestConfig(TestCase):
    """
    Test cases for the Config class, which handles the configuration settings
    for the AI and ensures it behaves as a singleton.
    """

    def setUp(self):
        """
        Set up the test environment by creating an instance of the Config class.
        """
        self.config = Config()

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
