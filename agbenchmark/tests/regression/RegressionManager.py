class RegressionManager:
    """Abstracts interaction with the regression tests file"""

    def __init__(self, filename: str):
        self.filename = filename
        self.load()

    def load(self) -> None:
        with open(self.filename, "r") as f:
            self.tests = f.readlines()

    def save(self) -> None:
        with open(self.filename, "w") as f:
            f.writelines(self.tests)

    def add_test(self, test_id) -> None:
        if f"{test_id}\n" not in self.tests:
            self.tests.append(f"{test_id}\n")

    def remove_test(self, test_id) -> None:
        if f"{test_id}\n" in self.tests:
            self.tests.remove(f"{test_id}\n")
