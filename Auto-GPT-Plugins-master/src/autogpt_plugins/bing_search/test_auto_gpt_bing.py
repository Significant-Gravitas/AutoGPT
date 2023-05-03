import os
import requests
import unittest
from typing import List
from . import AutoGPTBingSearch
from .bing_search import _bing_search


class TestAutoGPTBingSearch(unittest.TestCase):
    def setUp(self):
        os.environ["BING_API_KEY"] = "test_key"
        os.environ["SEARCH_ENGINE"] = "bing"
        self.plugin = AutoGPTBingSearch()

    def tearDown(self):
        os.environ.pop("SEARCH_ENGINE", None)
        os.environ.pop("BING_API_KEY", None)

    def test_bing_search(self):
        query = "test query"
        try:
            _bing_search(query)
        except requests.exceptions.HTTPError as e:
            self.assertEqual(e.response.status_code, 401)

    def test_pre_command(self):
        os.environ["SEARCH_ENGINE"] = "bing"
        self.plugin = AutoGPTBingSearch()

        command_name, arguments = self.plugin.pre_command(
            "google", {"query": "test query"}
        )
        self.assertEqual(command_name, "bing_search")
        self.assertEqual(arguments, {"query": "test query"})

    def test_can_handle_pre_command(self):
        self.assertTrue(self.plugin.can_handle_pre_command())

    def test_can_handle_post_prompt(self):
        self.assertTrue(self.plugin.can_handle_post_prompt())


if __name__ == "__main__":
    unittest.main()
