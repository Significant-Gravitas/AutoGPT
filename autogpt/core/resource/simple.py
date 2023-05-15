import logging
from typing import Any

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.resource.base import ResourceManager
from autogpt.core.resource.schema import ProviderBudget


class ResourceManagerConfiguration(SystemConfiguration):
    pass


class ResourceManagerSettings(SystemSettings):
    configuration = ResourceManagerConfiguration()


class SimpleResourceManager(ResourceManager, Configurable):
    defaults = ResourceManagerSettings(
        name="budget_manager",
        description="The budget manager is responsible for tracking resource usage.",
        configuration=ResourceManagerConfiguration(),
    )

    def __init__(
        self,
        configuration: ResourceManagerConfiguration,
        logger: logging.Logger,
        resources: dict[str, ProviderBudget],
    ):
        self._configuration = configuration
        self._logger = logger
        self._resources = resources

    def list_resources(self) -> list[str]:
        return list(self._resources)

    def get_total_budget(self, resource_name: str) -> float:
        return self._resources[resource_name].total_budget

    def get_total_cost(self, resource_name: str) -> float:
        return self._resources[resource_name].total_cost

    def get_remaining_budget(self, resource_name: str) -> float:
        return self._resources[resource_name].remaining_budget

    def get_resource_usage(self, resource_name: str) -> Any:
        return self._resources[resource_name].usage

    def update_resource_usage_and_cost(
        self,
        resource_name: str,
        *args,
        **kwargs,
    ) -> None:
        self._resources[resource_name].update_usage_and_cost(*args, **kwargs)

    def __repr__(self) -> str:
        return f"SimpleBudgetManager(resource_budgets={list(self._resources)})"
