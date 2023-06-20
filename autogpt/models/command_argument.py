class CommandArgument:
    def __init__(self, name: str, type: str, description: str, required: bool):
        self.name = name
        self.type = type
        self.description = description
        self.required = required

    def __repr__(self):
        return f"CommandArgument('{self.name}', '{self.type}', '{self.description}', {self.required})"
