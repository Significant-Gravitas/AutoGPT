class CommandArgument:
    def __init__(self, name: str, type: str, description: str, required: bool):
        self.name = name
        self.type = type
        self.description = description
        self.required = required
