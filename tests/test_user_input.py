import pytest
import pytest_mock
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from autogpt.utils import clean_input, ANSI_MAGENTA



class TestCleanInput:
    def test_no_color(self, mocker):
        # Mock the session.prompt() method to return a fixed input value
        mock_prompt = mocker.patch(
            "prompt_toolkit.shortcuts.PromptSession.prompt", return_value="test input"
        )

        prompt = clean_input("Input: ")
        expected_html = "Input: "

        assert prompt == "test input"
        mock_prompt.assert_called_with(expected_html)

    def test_clean_input_return_value(self, monkeypatch):
        color = ANSI_MAGENTA
        test_input = "test input"
        expected_html = test_input.strip()
        monkeypatch.setattr(
            "prompt_toolkit.shortcuts.PromptSession.prompt", lambda self, _: test_input
        )
        user_input = clean_input(color=color)
        assert user_input.strip() == expected_html

    def test_clean_input_keyboard_interrupt(self, mocker):
        mocker.patch("autogpt.utils.session.prompt", side_effect=KeyboardInterrupt)
        with pytest.raises(SystemExit) as excinfo:
            clean_input("Input: ")
        assert excinfo.value.code == 0

    def test_clean_input_eof_error(self, mocker):
        mocker.patch("autogpt.utils.session.prompt", side_effect=EOFError)
        result = clean_input("Input: ")
        assert result == ""

    def test_clean_input_formatted_text(self, mocker):
        mocker.patch(
            "autogpt.utils.session.prompt", return_value="formatted text input"
        )
        prompt = FormattedText([("fg:ansired", "Formatted text input: ")])
        result = clean_input(prompt=prompt)
        assert result == "formatted text input"
