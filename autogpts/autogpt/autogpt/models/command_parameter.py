import dataclasses


@dataclasses.dataclass
class CommandParameter:
    name: str
    type: str
    description: str
    required: bool

    def __repr__(self):
        return f"CommandParameter('{self.name}', '{self.type}', '{self.description}', {self.required})"
