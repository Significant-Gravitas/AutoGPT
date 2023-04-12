# JSON Utilities

This module provides various utilities for working with JSON strings.

## `extract_char_position()`

Extracts the character position from the `JSONDecodeError` message.

### Arguments

- `error_message` (`str`): The error message from the `JSONDecodeError` exception.

### Returns

- `int`: The character position.

### Example

```python
import json_utils

try:
    json.loads('{"key": "value,}')
except json.JSONDecodeError as e:
    char_position = json_utils.extract_char_position(str(e))
    print(f'The error occurred at character position {char_position}.')
```

## `add_quotes_to_property_names()`

Adds quotes to property names in a JSON string.

### Arguments

- `json_string` (`str`): The JSON string.

### Returns

- `str`: The JSON string with quotes added to property names.

### Example

```python
import json_utils

json_string = '{ key: "value" }'
fixed_json_string = json_utils.add_quotes_to_property_names(json_string)
print(fixed_json_string)  # Output: '{ "key": "value" }'
```

## `balance_braces()`

Balances the braces in a JSON string.

### Arguments

- `json_string` (`str`): The JSON string.

### Returns

- `str`: The JSON string with braces balanced.

### Example

```python
import json_utils

json_string = '{"key": "value", "nested": { "a": 1 }'
balanced_json_string = json_utils.balance_braces(json_string)
print(balanced_json_string)  # Output: '{"key": "value", "nested": { "a": 1 }}'
```

## `fix_invalid_escape()`

Fixes the `Invalid \\escape` error in a JSON string.

### Arguments

- `json_str` (`str`): The JSON string.
- `error_message` (`str`): The error message from the `JSONDecodeError` exception.

### Returns

- `str`: The corrected JSON string.

### Example

```python
import json_utils

json_string = '{ "key": "value\\", "nested": { "b": 2 } }'
try:
    json.loads(json_string)
except json.JSONDecodeError as e:
    fixed_json_string = json_utils.fix_invalid_escape(json_string, str(e))
    print(fixed_json_string)  # Output: '{ "key": "value", "nested": { "b": 2 } }'
```

## `correct_json()`

Corrects common JSON errors.

### Arguments

- `json_str` (`str`): The JSON string.

### Returns

- `str`: The corrected JSON string.

### Example

```python
import json_utils

json_string = '{key: "value", nested: {"a": 1, b: 2}}'
corrected_json_string = json_utils.correct_json(json_string)
print(corrected_json_string)
# Output: '{ "key": "value", "nested": { "a": 1, "b": 2 } }'
```