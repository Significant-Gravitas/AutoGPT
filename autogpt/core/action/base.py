import abc

from autogpt.core.action.schema import ActionRequirements, ActionResult


class Action(abc.ABC):
    """A class representing a command. Commands are actions which an agent can take.

    Attributes:
            name (str): The name of the command.
            description (str): A brief description of what the command does.
            signature (str): The signature of the function that theicommand executes. Defaults to None.
    """

    @abc.abstractmethod
    @property
    def name(self) -> str:
        ...

    @abc.abstractmethod
    @property
    def description(self) -> str:
        ...

    @abc.abstractmethod
    @property
    def arguments(self) -> list[str]:
        ...

    @property
    def signature(self) -> str:
        return " ".join(self.arguments) if self.arguments else ""

    @abc.abstractmethod
    @property
    def requirements(self) -> ActionRequirements:
        ...

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> ActionResult:
        ...

    def __str__(self) -> str:
        return f"{self.name}: {self.description}, args: {self.signature}"


class ActionRegistry(abc.ABC):
    def register_action(self, action: Action) -> None:
        ...

    def list_actions(self) -> list[str]:
        ...

    def get_action(self, action_name: str) -> Action:
        ...

    def act(self, action_name: str, **kwargs) -> ActionResult:
        ...
