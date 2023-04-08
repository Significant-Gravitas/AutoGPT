import json
import re
from typing import Any, List, Type

from pydantic import BaseModel, ValidationError

PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""


def parse_pydantic_object(text: str, pydantic_object: Type[BaseModel]):
    try:
        pattern = rf"{pydantic_object.__name__}\s*:\s*"
        match = re.search(pattern, text, re.MULTILINE)

        if not match:
            raise ValueError(f"{pydantic_object.__name__} not found in text")

        start = match.end()

        # Initialize the JSONDecoder
        decoder = json.JSONDecoder()

        # Find the next valid JSON object
        while start < len(text):
            try:
                json_obj, _ = decoder.raw_decode(text, start)
                # Check if the JSON object has the required keys
                if all(
                    key in json_obj for key in pydantic_object.schema()["properties"]
                ):
                    break
                else:
                    # Update the start position to continue searching
                    start += 1
            except json.JSONDecodeError:
                start += 1

        # Parse the JSON object and return the pydantic_object
        return pydantic_object.parse_obj(json_obj)

    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        name = pydantic_object.__name__
        msg = f"Failed to parse {name} from completion {text}. Got: {e}"
        print("DEBUG, msg:" + msg)
        raise Exception(msg)


def get_format_instructions(objects: List[Type[BaseModel]]) -> str:
    combined_schema = {}

    for obj in objects:
        schema = obj.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]

        # Add object name to the schema
        object_name = obj.__name__
        combined_schema[object_name] = reduced_schema["properties"]

    # Ensure json in context is well-formed with double quotes.
    json_schema = json.dumps(combined_schema)

    return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=json_schema)
