import unittest
import coverage

# Define the source directories that you want to measure coverage for
source_dirs = ["src"]

# Start the coverage measurement
cov = coverage.Coverage(source=["src"])
cov.start()

# Discover and run tests
loader = unittest.TestLoader()
suite = loader.discover("tests")
runner = unittest.TextTestRunner()
result = runner.run(suite)

# Stop the coverage measurement
cov.stop()
cov.save()

# Generate and display the coverage report
cov.report()
