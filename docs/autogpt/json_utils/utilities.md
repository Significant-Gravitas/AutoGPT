## `json_utils.py`

This module provides utilities that help with JSON processing.

### `extract_char_position(error_message: str) -> int`
Extracts the character position from the error message of a JSONDecodeError.

**Arguments**:
- `error_message` (str): The error message from the JSONDecodeError.

**Returns**:
- `int`: The character position.

### `validate_json(json_object: object, schema_name: object) -> object`
Validates a `json_object` against a `schema_name` using the `jsonschema.Draft7Validator`.

**Arguments**:
- `json_object` (object): The JSON object to be validated.
- `schema_name` (object): The name of the JSON schema file to be used for validation.

**Returns**:
- `object`: The validated JSON object.

#### Example Usage:

```python
import json_utils

# Example JSON object
example_json = {
    "name": "John Doe",
    "age": 31,
    "city": "New York"
}

# Example schema name (stored as a JSON file)
example_schema = "example_schema"

valid_json = json_utils.validate_json(example_json, example_schema)
```