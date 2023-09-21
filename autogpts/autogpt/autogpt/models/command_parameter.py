import dataclasses
import enum

from autogpt.core.utils.json_schema import JSONSchema


@dataclasses.dataclass
class CommandParameter:
    name: str
    type: JSONSchema.Type | enum.Enum
    description: str
    required: bool

    def __repr__(self):
        return f"CommandParameter('{self.name}', '{self.type}', '{self.description}', {self.required})"
