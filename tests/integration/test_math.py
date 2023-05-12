import pytest

import autogpt.commands.execute_code as sut  # system under testing
from autogpt.commands.math import evaluate_expression


def test_evaluate_expression():
    # Test case 1: Valid expression
    expr = "2 + 3 * 4"
    expected_result = 14
    assert evaluate_expression(expr) == expected_result

    # Test case 2: Another valid expression
    expr = "(5 - 2) / 3"
    expected_result = 1
    assert evaluate_expression(expr) == expected_result

    # Test case 3: Invalid expression
    #expr = "2 / 0"  # Division by zero
    #with pytest.raises(ZeroDivisionError) as exc_info:
    #    evaluate_expression(expr)
    #assert str(exc_info.type) == "<class 'ZeroDivisionError'>"
