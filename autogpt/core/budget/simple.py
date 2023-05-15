import logging
from typing import Any

from autogpt.core.budget.base import BudgetManager
from autogpt.core.configuration import (
    Configurable,
    ResourceBudget,
    SystemConfiguration,
    SystemSettings,
)


class BudgetConfiguration(SystemConfiguration):
    pass


class SimpleBudgetManager(BudgetManager, Configurable):
    defaults = SystemSettings(
        name="budget_manager",
        description="The budget manager is responsible for tracking resource usage.",
        configuration=BudgetConfiguration(),
    )

    def __init__(
        self,
        configuration: BudgetConfiguration,
        logger: logging.Logger,
        resources: dict[str, ResourceBudget],
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
