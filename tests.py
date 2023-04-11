import unittest

if __name__ == "__main__":
    # Load all tests from the 'scripts/tests' package
    suite = unittest.defaultTestLoader.discover('scripts/tests')
    
    # Run the tests
    unittest.TextTestRunner().run(suite)
