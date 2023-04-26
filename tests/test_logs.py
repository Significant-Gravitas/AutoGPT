import io
import sys
from unittest.mock import MagicMock

import pytest

from autogpt.api_manager import ApiManager
from autogpt.logs import _remaining_budget_description, print_assistant_thoughts


@pytest.fixture
def api_manager():
    # Mocking an instance of APIManager with overrided get_total_budget() and get_total_cost() methods
    class _MockAPIManager(ApiManager):
        def get_total_budget(self):
            return 100

        def get_total_cost(self):
            return 20

    return _MockAPIManager()


def test_remaining_budget_description_positive(api_manager, monkeypatch):
    # Mocking the 'api_manager' import in the 'logs' module
    monkeypatch.setattr("autogpt.api_manager.api_manager", api_manager)

    result = _remaining_budget_description()
    assert result == "$80 remaining from $100."


def test_print_assistant_thoughts_without_budget(capsys, monkeypatch):
    # Set total budget to 0 and total cost to 13
    api_manager.get_total_budget = MagicMock(return_value=0)
    api_manager.get_total_cost = MagicMock(return_value=13)

    # Mock the 'api_manager' import in the 'logs' module
    monkeypatch.setattr("autogpt.api_manager.api_manager", api_manager)

    # Redirect stdout to capture printed output
    sys.stdout = io.StringIO()

    print_assistant_thoughts("AI", {"thoughts": {}})
    captured = sys.stdout.getvalue()

    # Reset the stdout redirection
    sys.stdout = sys.__stdout__

    # Assert that no budget information is printed when total budget is 0
    assert "BUDGET:" not in captured


def test_print_assistant_thoughts_with_budget(capsys, monkeypatch):
    # Set total budget to 13 and total cost to 5
    api_manager.get_total_budget = MagicMock(return_value=13)
    api_manager.get_total_cost = MagicMock(return_value=5)

    # Mock the 'api_manager' import in the 'logs' module
    monkeypatch.setattr("autogpt.api_manager.api_manager", api_manager)

    # Redirect stdout to capture printed output
    sys.stdout = io.StringIO()

    print_assistant_thoughts("AI", {"thoughts": {}})
    captured = sys.stdout.getvalue()

    # Reset the stdout redirection
    sys.stdout = sys.__stdout__

    # Assert that budget information is printed when total budget is 13
    assert "BUDGET:" in captured
    assert "$8 remaining from $13." in captured
