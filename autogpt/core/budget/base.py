import abc
from typing import Any


class ResourceBudget(abc.ABC):
    """Representation of the budget of a particular resource."""

    configuration_defaults = {}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @property
    @abc.abstractmethod
    def total_budget(self) -> float:
        """Total budget for the resource."""
        ...

    @property
    @abc.abstractmethod
    def total_cost(self) -> float:
        """Total cost of all prior resource usage."""
        ...

    @property
    @abc.abstractmethod
    def remaining_budget(self) -> float:
        """Remaining budget for the resource."""
        ...

    @property
    @abc.abstractmethod
    def usage(self) -> Any:
        """Total usage of the resource."""
        ...

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> None:
        """Update the usage and cost of the resource."""
        ...

    @abc.abstractmethod
    def get_resource_budget_prompt(self) -> str:
        """Get the prompt to be used for the resource budget."""
        # TODO: does this belong here?  Not really sure...
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...


class BudgetManager(abc.ABC):
    """The BudgetManager class is a manager for constrained resources."""

    configuration_defaults = {"budget_manager": {}}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def list_resources(self) -> list[str]:
        """List the resources that are being managed."""
        ...

    @abc.abstractmethod
    def get_total_budget(self, resource_name: str) -> float:
        """Get the total budget for a resource."""
        ...

    @abc.abstractmethod
    def get_total_cost(self, resource_name: str) -> float:
        """Get the total cost of a resource."""
        ...

    @abc.abstractmethod
    def get_remaining_budget(self, resource_name: str) -> float:
        """Get the remaining budget for a resource."""
        ...

    @abc.abstractmethod
    def get_resource_usage(self, resource_name: str) -> Any:
        """Get the usage of a resource."""
        ...

    @abc.abstractmethod
    def update_resource_usage_and_cost(
        self, resource_name: str, *args, **kwargs
    ) -> None:
        """Update the usage and cost of a resource."""
        ...

    @abc.abstractmethod
    def get_resource_budget_prompt(self, resource_name: str) -> str:
        """Get the prompt to be used for a resource budget."""
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        ...
