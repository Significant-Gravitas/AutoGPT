import dataclasses

from forge.json.model import JSONSchema


@dataclasses.dataclass
class CommandParameter:
    name: str
    spec: JSONSchema

    def __repr__(self):
        return "CommandParameter('%s', '%s', '%s', %s)" % (
            self.name,
            self.spec.type,
            self.spec.description,
            self.spec.required,
        )
