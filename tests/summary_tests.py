import sys
import tempfile
import unittest
from unittest.mock import patch

from pathlib import Path

# Ugly workaround required to import modules from parent directory
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from conversation_summary.summary import Summary, SummaryUtils


# Utility functions for testing
def mock_send_message(message: str) -> tuple:
    return None, "Generated summary text."


def mock_create_agent_callback(next_agent_key: int, message: str, gpt_agent_model: str) -> tuple:
    return None, "Generated summary text."


class TestSummary(unittest.TestCase):

    def setUp(self):
        """Set up the test environment by initializing a Summary instance."""
        self.step_summary_prompt = "Please provide a step-by-step summary."
        self.final_summary_prompt = "Please provide a final summary."
        self.ai_name = "TestAI"
        self.summary = Summary(self.step_summary_prompt, self.final_summary_prompt, self.ai_name)

    def test_init(self):
        """Test the initialization of a Summary instance."""
        self.assertIsInstance(self.summary, Summary)
        self.assertEqual(self.summary.step_counter, 1)
        self.assertEqual(self.summary.step_summarization_prompt, self.step_summary_prompt)
        self.assertEqual(self.summary.final_summarization_prompt, self.final_summary_prompt)

    def test_increment_step_counter(self):
        """Test if the step counter is incremented correctly."""
        initial_step_counter = self.summary.step_counter
        self.summary.increment_step_counter()
        self.assertEqual(self.summary.step_counter, initial_step_counter + 1)

    def test_print_step_summary_to_console(self):
        """Test if the step summary is printed to the console."""
        step_summary = "Step summary text."
        with patch('builtins.print') as mock_print:
            self.summary.print_step_summary_to_console(step_summary)
        mock_print.assert_called_once()

    def test_print_final_summary_to_console(self):
        """Test if the final summary is printed to the console."""
        final_summary = "Final summary text."
        with patch('builtins.print') as mock_print:
            Summary.print_final_summary_to_console(final_summary)
        mock_print.assert_called_once()

    def test_append_step_summary_to_file(self):
        """Test if the step summary and result are appended to the summary file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_file = Path(temp_dir) / "temp_summary.txt"
            self.summary.summary_filename = summary_file
            self.summary.append_summarized_step_to_file("Step summary text.", "Step result text.")
            self.assertTrue(summary_file.exists())
            with open(summary_file, "r") as f:
                content = f.read()
            self.assertIn("Step summary text.", content)
            self.assertIn("Step result text.", content)

    def test_format_final_summary_prompt(self):
        """Test if the final summary prompt is formatted correctly."""
        final_summary_prompt = "Please provide a summary of the following content with bullet points: {}"
        chunk = "Example text."
        formatted_prompt = Summary._format_final_summary_prompt(final_summary_prompt, chunk)
        expected_prompt = 'Please provide a summary of the following content with bullet points: Example text.'
        self.assertEqual(formatted_prompt, expected_prompt)

        # Test with incorrect final summary prompt
        incorrect_prompt = "Please provide a summary of the following content with bullet points:"
        formatted_prompt = Summary._format_final_summary_prompt(incorrect_prompt, chunk)
        expected_error = "The prompt string must contain one instance of '{}' which will include the chunk of text. " \
                         "Please update your FINAL_SUMMARY_PROMPT variable in the .env file."
        self.assertEqual(formatted_prompt, expected_error)


class TestSummaryUtils(unittest.TestCase):

    def test_get_step_summary_from_assistant_reply(self):
        """Test if the step summary is extracted correctly from the assistant's reply."""
        assistant_reply = '{"response": "Response text.", "summary": "Summary text."}'
        summary = SummaryUtils.get_step_summary_from_assistant_reply(
            assistant_reply,
            fix_and_parse_json=lambda x: x,
            print_to_console=lambda *args, **kwargs: None
        )
        self.assertEqual(summary, "Summary text.")


if __name__ == "__main__":
    unittest.main()
