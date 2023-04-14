import unittest

if __name__ == "__main__":
    # Load all tests from the 'autogpt/tests' package
    suite = unittest.defaultTestLoader.discover("autogpt/tests")

    # Run the tests
    unittest.TextTestRunner().run(suite)
