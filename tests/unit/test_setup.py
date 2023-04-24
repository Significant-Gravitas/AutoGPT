import unittest
from io import StringIO
from unittest.mock import patch

from autogpt.config.ai_config import AIConfig
from autogpt.setup import (
    generate_aiconfig_automatic,
    generate_aiconfig_manual,
    prompt_user,
)
from tests.utils import requires_api_key


class TestAutoGPT(unittest.TestCase):
    @requires_api_key("OPENAI_API_KEY")
    def test_generate_aiconfig_automatic_default(self):
        user_inputs = [""]
        with patch("builtins.input", side_effect=user_inputs):
            ai_config = prompt_user()

        self.assertIsInstance(ai_config, AIConfig)
        self.assertIsNotNone(ai_config.ai_name)
        self.assertIsNotNone(ai_config.ai_role)
        self.assertGreaterEqual(len(ai_config.ai_goals), 1)
        self.assertLessEqual(len(ai_config.ai_goals), 5)

    @requires_api_key("OPENAI_API_KEY")
    def test_generate_aiconfig_automatic_typical(self):
        user_prompt = "Help me create a rock opera about cybernetic giraffes"
        ai_config = generate_aiconfig_automatic(user_prompt)

        self.assertIsInstance(ai_config, AIConfig)
        self.assertIsNotNone(ai_config.ai_name)
        self.assertIsNotNone(ai_config.ai_role)
        self.assertGreaterEqual(len(ai_config.ai_goals), 1)
        self.assertLessEqual(len(ai_config.ai_goals), 5)

    @requires_api_key("OPENAI_API_KEY")
    def test_generate_aiconfig_automatic_fallback(self):
        user_inputs = [
            "T&GF£OIBECC()!*",
            "Chef-GPT",
            "an AI designed to browse bake a cake.",
            "Purchase ingredients",
            "Bake a cake",
            "",
            "",
        ]
        with patch("builtins.input", side_effect=user_inputs):
            ai_config = prompt_user()

        self.assertIsInstance(ai_config, AIConfig)
        self.assertEqual(ai_config.ai_name, "Chef-GPT")
        self.assertEqual(ai_config.ai_role, "an AI designed to browse bake a cake.")
        self.assertEqual(ai_config.ai_goals, ["Purchase ingredients", "Bake a cake"])

    @requires_api_key("OPENAI_API_KEY")
    def test_prompt_user_manual_mode(self):
        user_inputs = [
            "--manual",
            "Chef-GPT",
            "an AI designed to browse bake a cake.",
            "Purchase ingredients",
            "Bake a cake",
            "",
            "",
        ]
        with patch("builtins.input", side_effect=user_inputs):
            ai_config = prompt_user()

        self.assertIsInstance(ai_config, AIConfig)
        self.assertEqual(ai_config.ai_name, "Chef-GPT")
        self.assertEqual(ai_config.ai_role, "an AI designed to browse bake a cake.")
        self.assertEqual(ai_config.ai_goals, ["Purchase ingredients", "Bake a cake"])

    @requires_api_key("OPENAI_API_KEY")
    def test_prompt_user_manual_mode_default(self):
        user_inputs = [
            "--manual",
            "",
            "",
            "",
            "",
        ]
        with patch("builtins.input", side_effect=user_inputs):
            ai_config = prompt_user()

        default_ai_name = "Entrepreneur-GPT"
        default_ai_role = (
            "an AI designed to autonomously develop and run businesses with the"
            " sole goal of increasing your net worth."
        )
        default_ai_goals = [
            "Increase net worth",
            "Grow Twitter Account",
            "Develop and manage multiple businesses autonomously",
        ]

        self.assertIsInstance(ai_config, AIConfig)
        self.assertEqual(ai_config.ai_name, default_ai_name)
        self.assertEqual(ai_config.ai_role, default_ai_role)
        self.assertEqual(ai_config.ai_goals, default_ai_goals)


if __name__ == "__main__":
    unittest.main()
