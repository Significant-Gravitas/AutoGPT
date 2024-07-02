import enum
from textwrap import indent
from typing import Optional, overload

from jsonschema import Draft7Validator, ValidationError
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
        definitions = schema.get("$defs", {})
        schema = _resolve_type_refs_in_schema(schema, definitions)

        return JSONSchema(
            description=schema.get("description"),
            type=schema["type"],
            enum=schema.get("enum"),
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

    def validate_object(self, object: object) -> tuple[bool, list[ValidationError]]:
        """
        Validates an object or a value against the JSONSchema.

        Params:
            object: The value/object to validate.
            schema (JSONSchema): The JSONSchema to validate against.

        Returns:
            bool: Indicates whether the given value or object is valid for the schema.
            list[ValidationError]: The issues with the value or object (if any).
        """
        validator = Draft7Validator(self.to_dict())

        if errors := sorted(validator.iter_errors(object), key=lambda e: e.path):
            return False, errors

        return True, []

    def to_typescript_object_interface(self, interface_name: str = "") -> str:
        if self.type != JSONSchema.Type.OBJECT:
            raise NotImplementedError("Only `object` schemas are supported")

        if self.properties:
            attributes: list[str] = []
            for name, property in self.properties.items():
                if property.description:
                    attributes.append(f"// {property.description}")
                attributes.append(f"{name}: {property.typescript_type};")
            attributes_string = "\n".join(attributes)
        else:
            attributes_string = "[key: string]: any"

        return (
            f"interface {interface_name} " if interface_name else ""
        ) + f"{{\n{indent(attributes_string, '  ')}\n}}"

    @property
    def typescript_type(self) -> str:
        if not self.type:
            return "any"
        if self.type == JSONSchema.Type.BOOLEAN:
            return "boolean"
        if self.type in {JSONSchema.Type.INTEGER, JSONSchema.Type.NUMBER}:
            return "number"
        if self.type == JSONSchema.Type.STRING:
            return "string"
        if self.type == JSONSchema.Type.ARRAY:
            return f"Array<{self.items.typescript_type}>" if self.items else "Array"
        if self.type == JSONSchema.Type.OBJECT:
            if not self.properties:
                return "Record<string, any>"
            return self.to_typescript_object_interface()
        if self.enum:
            return " | ".join(repr(v) for v in self.enum)

        raise NotImplementedError(
            f"JSONSchema.typescript_type does not support Type.{self.type.name} yet"
        )


@overload
def _resolve_type_refs_in_schema(schema: dict, definitions: dict) -> dict:
    ...


@overload
def _resolve_type_refs_in_schema(schema: list, definitions: dict) -> list:
    ...


def _resolve_type_refs_in_schema(schema: dict | list, definitions: dict) -> dict | list:
    """
    Recursively resolve type $refs in the JSON schema with their definitions.
    """
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"].split("/")[2:]  # Split and remove '#/definitions'
            ref_value = definitions
            for key in ref_path:
                ref_value = ref_value[key]
            return _resolve_type_refs_in_schema(ref_value, definitions)
        else:
            return {
                k: _resolve_type_refs_in_schema(v, definitions)
                for k, v in schema.items()
            }
    elif isinstance(schema, list):
        return [_resolve_type_refs_in_schema(item, definitions) for item in schema]
    else:
        return schema
