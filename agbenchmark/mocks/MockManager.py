import sys
import agbenchmark.mocks.tests.basic_mocks as basic_mocks
import agbenchmark.mocks.tests.retrieval_mocks as retrieval_mocks


class MockManager:
    def __init__(self, task: str):
        self.task = task
        self.workspace = "agbenchmark/mocks/workspace"
        self.modules = [basic_mocks, retrieval_mocks]

    def delegate(self, mock_function_name, *args, **kwargs):
        if hasattr(self, mock_function_name):
            # Check if the mock function is an attribute of this class
            getattr(self, mock_function_name)(*args, **kwargs)
        elif mock_function_name in globals():
            # Check if the function is imported in the file
            func = globals()[mock_function_name]
            func(self.task, self.workspace, *args, **kwargs)
        elif len(self.modules) > 0:
            # checks if function is in imported modules
            for module in self.modules:
                if hasattr(module, mock_function_name):
                    func = getattr(module, mock_function_name)
                    func(self.task, self.workspace, *args, **kwargs)
                    return
        else:
            raise ValueError(f"No such mock: {mock_function_name}")
