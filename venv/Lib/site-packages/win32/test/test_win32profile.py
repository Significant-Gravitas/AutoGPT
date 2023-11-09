"""Test win32profile"""
import os
import unittest

import win32profile


class Tester(unittest.TestCase):
    def test_environment(self):
        os.environ["FOO"] = "bar=baz"
        env = win32profile.GetEnvironmentStrings()
        assert "FOO" in env
        assert env["FOO"] == "bar=baz"
        assert os.environ["FOO"] == "bar=baz"


if __name__ == "__main__":
    unittest.main()
