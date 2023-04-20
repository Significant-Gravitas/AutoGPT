# JSON Utils Module

This module contains functions to fix JSON strings using general programmatic approaches, suitable for addressing common JSON formatting issues.

## Functions

### `fix_invalid_escape`

Fix invalid escape sequences in JSON strings.

```python
fix_invalid_escape(json_to_load: str, error_message: str) -> str
```

**Arguments:**
- `json_to_load` (str): The JSON string.
- `error_message` (str): The error message from the `JSONDecodeError` exception.

**Returns:**
- `str`: The JSON string with invalid escape sequences fixed.

### `balance_braces`

Balance the braces in a JSON string.

```python
balance_braces(json_string: str) -> Optional[str]
```

**Arguments:**
- `json_string` (str): The JSON string.

**Returns:**
- `str`: The JSON string with braces balanced.

### `add_quotes_to_property_names`

Add quotes to property names in a JSON string.

```python
add_quotes_to_property_names(json_string: str) -> str
```

**Arguments:**
- `json_string` (str): The JSON string.

**Returns:**
- `str`: The JSON string with quotes added to property names.

### `correct_json`

Correct common JSON errors.

```python
correct_json(json_to_load: str) -> str
```

**Arguments:**
- `json_to_load` (str): The JSON string.

**Returns:**
- `str`: The corrected JSON string.