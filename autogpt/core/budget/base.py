import abc


class BudgetManager(abc.ABC):
    """
    The BudgetManager class is a manager for constrained resources.
    Initially only supports monetary budgets, but could be extended to support time, memory, etc.
    """

    @abc.abstractmethod
    def __init__(self, budget: float = 0.00):
        pass

    @abc.abstractmethod
    def set_budget(self, budget: float = 0.00) -> None:
        pass

    @abc.abstractmethod
    def get_budget(self) -> float:
        pass

    @abc.abstractmethod
    def get_spend(self) -> float:
        pass

    @abc.abstractmethod
    def record_cost(self, amount: float = 0.00) -> None:
        pass


class Budget(abc.ABC):
    # TODO: Not used yet
    pass


class BudgetManagerConcrete(BudgetManager):
    configuration_defaults = {
        "budget_manager": {
            "budget": 0.00  # This means the agent has an infinite budget
        }
    }

    initial_budget: float = 0.00
    remaining_budget: float = 0.00

    def __init__(self, budget: float = 0.00):
        self.initial_budget = budget
        self.set_budget(budget)

    def set_budget(self, budget: float = 0.00) -> None:
        if budget < 0.00:
            raise ValueError("Budget cannot be negative.")
        self.remaining_budget = budget

    def get_budget(self) -> float:
        return self.remaining_budget

    def get_spend(self) -> float:
        return self.initial_budget - self.remaining_budget

    def record_cost(self, amount: float = 0.00) -> None:
        if amount > self.remaining_budget:
            raise ValueError("Cost exceeds remaining budget.")
        self.remaining_budget -= amount
