import dataclasses


@dataclasses.dataclass
class CommandArgument:
    name: str
    type: str
    description: str
    required: bool

    def __repr__(self):
        return f"CommandArgument('{self.name}', '{self.type}', '{self.description}', {self.required})"
