from forge.models.json_schema import JSONSchema
from pydantic import BaseModel


class CommandParameter(BaseModel):
    name: str
    spec: JSONSchema

    def __repr__(self):
        return "CommandParameter('%s', '%s', '%s', %s)" % (
            self.name,
            self.spec.type,
            self.spec.description,
            self.spec.required,
        )
