# `Code Improver` Module

This module contains three functions- `evaluate_code`, `improve_code`, and `write_tests`.

## `evaluate_code`

```python
def evaluate_code(code: str) -> List[str]:
    """
    A function that takes in a string and returns a response from create chat completion api call.

    Parameters:
        code (str): Code to be evaluated.

    Returns:
        list: A list of suggestions to improve the code.
    """
```

This function takes a `code` string as an input and returns a list of suggestions to improve the code as output. It analyzes the given code and generates the suggestions.

## `improve_code`

```python
def improve_code(suggestions: List[str], code: str) -> str:
    """
    A function that takes in code and suggestions and returns a response from create chat completion api call.

    Parameters:
        suggestions (List): A list of suggestions around what needs to be improved.
        code (str): Code to be improved.

    Returns:
        str: Improved code in response.
    """
```

This function takes a list of `suggestions` and `code` as inputs and returns an improved version of the `code` as output. It makes changes only based on the suggestions provided.

## `write_tests`

```python
def write_tests(code: str, focus: List[str]) -> str:
    """
    A function that takes in code and focus topics and returns a response from create chat completion api call.

    Parameters:
        focus (List): A list of suggestions around what needs to be improved.
        code (str): Code for test cases to be generated against.

    Returns:
        str: Test cases for the submitted code in response.
    """
```

This function takes `code` and a `focus` list as inputs and returns generated test cases as output. It generates test cases for the given `code`, focusing on specific areas if required.

### Example 

```python
code = "def foo(a, b):\n    return a+b"

# Obtain suggestions for the code with `evaluate_code`
suggestions = evaluate_code(code)

# Improve code as suggested with `improve_code`
improved_code = improve_code(suggestions, code)

# Generate the test cases using `write_tests`
test_cases = write_tests(improved_code, ["edge cases", "boundary conditions"])
```