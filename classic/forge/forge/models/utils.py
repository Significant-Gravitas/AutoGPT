from abc import ABC, abstractmethod

from pydantic import BaseModel


class ModelWithSummary(BaseModel, ABC):
    @abstractmethod
    def summary(self) -> str:
        """Should produce a human readable summary of the model content."""
        pass
