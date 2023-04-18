import unittest
import pytest
import logging

from unittest.mock import MagicMock

from autogpt.commands.validators import CommandsValidator
from autogpt.logs import logger

@pytest.mark.integration_test
class TestCommandsValidator(unittest.TestCase):

    def setUp(self):
        logger.set_level(logging.DEBUG)

    def test_browse_site_command_valid_url(self):
        cv = CommandsValidator("browse_site", {"url": "https://www.example.com"})
        cv.validate_browse_site_command()
        cv=CommandsValidator("browse_site", {"url": "http://www.example.com"})
        cv.validate_browse_site_command()
        # assert that no exception was raised

    def test_browse_site_command_invalid_url(self):
        cv = CommandsValidator("browse_site", {"url": "not_a_valid_url"})
        with self.assertRaises(Exception) as cm:
            cv.validate_browse_site_command()
        error_message = str(cm.exception)
        self.assertEqual(error_message, "argument url must be a valid url")



    def test_browse_site_command_invalid_key(self):
        cv=CommandsValidator("browse_site", {"invalid_key": "not_a_valid_url"})
        with self.assertRaises(Exception) as cm:
            cv.validate_browse_site_command()
        error_message = str(cm.exception)
        self.assertEqual(error_message, "argument url must be a valid url")

    def test_validate_command_browse_site(self):
        cv = CommandsValidator("browse_site", {"url": "https://www.example.com"})
        cv.validate_command()
        # assert that no exception was raised

    def test_validate_command_skip_validation(self, caplog):
        logger.set_level(logging.DEBUG)
        with caplog.at_leve(logging.DEBUG):
            cv = CommandsValidator("other_command", {})
            cv.validate_command()
            assert "skipping validation for command: other_command" in caplog.text


