import json


class RegressionManager:
    """Abstracts interaction with the regression tests file"""

    def __init__(self, filename: str):
        self.filename = filename
        self.load()

    def load(self) -> None:
        try:
            with open(self.filename, "r") as f:
                self.tests = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            self.tests = {}

    def save(self) -> None:
        with open(self.filename, "w") as f:
            json.dump(self.tests, f, indent=4)

    def add_test(self, test_name: str, test_details: dict) -> None:
        self.tests[test_name] = test_details
        self.save()

    def remove_test(self, test_name: str) -> None:
        if test_name in self.tests:
            del self.tests[test_name]
            self.save()
