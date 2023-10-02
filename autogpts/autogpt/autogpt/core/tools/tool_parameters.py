import dataclasses

from autogpt.core.utils.json_schema import JSONSchema


@dataclasses.dataclass
class ToolParameter:
    name: str
    spec: JSONSchema

    def __repr__(self):
        return f"ToolParameter('{self.name}', '{self.spec.type}', '{self.spec.description}', {self.spec.required})"
    
    def dump(self) ->dict[str,JSONSchema]: 
       return {self.name : self.spec  }
