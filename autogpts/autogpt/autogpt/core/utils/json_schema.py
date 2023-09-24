import enum
import json
from logging import Logger
from typing import Literal, Optional

from jsonschema import Draft7Validator
from pydantic import BaseModel


class JSONSchema(BaseModel):
    class Type(str, enum.Enum):
        STRING = "string"
        ARRAY = "array"
        OBJECT = "object"
        NUMBER = "number"
        INTEGER = "integer"
        BOOLEAN = "boolean"

    # TODO: add docstrings
    description: Optional[str] = None
    type: Optional[Type] = None
    enum: Optional[list] = None
    required: bool = False
    items: Optional["JSONSchema"] = None
    properties: Optional[dict[str, "JSONSchema"]] = None
    minimum: Optional[int | float] = None
    maximum: Optional[int | float] = None
    minItems: Optional[int] = None
    maxItems: Optional[int] = None

    def to_dict(self) -> dict:
        schema: dict = {
            "type": self.type.value if self.type else None,
            "description": self.description,
        }
        if self.type == "array":
            if self.items:
                schema["items"] = self.items.to_dict()
            schema["minItems"] = self.minItems
            schema["maxItems"] = self.maxItems
        elif self.type == "object":
            if self.properties:
                schema["properties"] = {
                    name: prop.to_dict() for name, prop in self.properties.items()
                }
                schema["required"] = [
                    name for name, prop in self.properties.items() if prop.required
                ]
        elif self.enum:
            schema["enum"] = self.enum
        else:
            schema["minumum"] = self.minimum
            schema["maximum"] = self.maximum

        schema = {k: v for k, v in schema.items() if v is not None}

        return schema

    @staticmethod
    def from_dict(schema: dict) -> "JSONSchema":
        return JSONSchema(
            description=schema.get("description"),
            type=schema["type"],
            enum=schema["enum"] if "enum" in schema else None,
            items=JSONSchema.from_dict(schema["items"]) if "items" in schema else None,
            properties=JSONSchema.parse_properties(schema)
            if schema["type"] == "object"
            else None,
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            minItems=schema.get("minItems"),
            maxItems=schema.get("maxItems"),
        )

    @staticmethod
    def parse_properties(schema_node: dict) -> dict[str, "JSONSchema"]:
        properties = (
            {k: JSONSchema.from_dict(v) for k, v in schema_node["properties"].items()}
            if "properties" in schema_node
            else {}
        )
        if "required" in schema_node:
            for k, v in properties.items():
                v.required = k in schema_node["required"]
        return properties

    def validate_object(
        self, object: object, logger: Logger
    ) -> tuple[Literal[True], None] | tuple[Literal[False], list]:
        """
        Validates a dictionary object against the JSONSchema.

        Params:
            object: The dictionary object to validate.
            schema (JSONSchema): The JSONSchema to validate against.

        Returns:
            tuple: A tuple where the first element is a boolean indicating whether the object is valid or not,
                and the second element is a list of errors found in the object, or None if the object is valid.
        """
        validator = Draft7Validator(self.to_dict())

        if errors := sorted(validator.iter_errors(object), key=lambda e: e.path):
            for error in errors:
                logger.debug(f"JSON Validation Error: {error}")

            logger.error(json.dumps(object, indent=4))
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
            return False, errors

        logger.debug("The JSON object is valid.")

        return True, None
