import abc
from typing import Any

from autogpt.core.configuration import SystemConfiguration


class BudgetManager(abc.ABC):
    """The BudgetManager class is a manager for constrained resources."""

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
    def __repr__(self) -> str:
        ...
