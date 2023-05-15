import typing

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    # These are all cyclic imports, so we need to use strings.

    from autogpt.core.model import (
        EmbeddingModelConfiguration,
        LanguageModelConfiguration,
        Prov,
    )
    from autogpt.core.planning import PlannerConfiguration
    from autogpt.core.workspace import WorkspaceConfiguration


class AgentConfiguration(BaseModel):
    embedding_mode: "EmbeddingModelConfiguration"
    language_model: "LanguageModelConfiguration"
    model_provider: dict[str, "OpenAIConfiguration"]
    planning: "PlannerConfiguration"
    workspace: "WorkspaceConfiguration"


#
# class AgentConfiguration(abc.ABC):
#     @abc.abstractmethod
#     def __init__(self, *arg, **kwargs):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def system(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def budget_manager(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def command_registry(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def credentials(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def language_model(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def memory_backend(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def message_broker(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def planner(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def plugin_manager(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def workspace(self):
#         pass
#
#     @abc.abstractmethod
#     def to_dict(self):
#         pass
#
#     @abc.abstractmethod
#     def from_dict(self):
#         pass
#
#     @abc.abstractmethod
#     def __repr__(self):
#         pass
