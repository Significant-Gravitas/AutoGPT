import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.wait import WebDriverWait

from autogpt.commands.web_selenium import scrape_text_with_selenium
from autogpt.config import Config
from autogpt.workspace import Workspace


class TestSelenium(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        workspace_path = os.path.join(os.path.dirname(__file__), "workspace")
        self.workspace_path = Workspace.make_workspace(workspace_path)
        self.config.workspace_path = workspace_path
        self.workspace = Workspace(workspace_path, restrict_to_workspace=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_path)

    def test_selenium_chromium_binary_location(self):
        self.config.chromium_binary_location = "custom_chrome.exe"

        with self.assertRaises(WebDriverException) as excep:
            result = scrape_text_with_selenium("http://localhost")

        self.assertTrue(
            "no chrome binary at " + self.config.chromium_binary_location
            in str(excep.exception)
        )
