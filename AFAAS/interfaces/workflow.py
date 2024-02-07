from typing import ClassVar

from pydantic import BaseModel, Field


class BaseWorkflow(BaseModel):  # NOTE: Might be merged with Pipeline
    name: ClassVar[str]
    description: ClassVar[str]
    examples: ClassVar[list[str]] = Field(
        ..., description="Examples of tasks that would fit this workflow.", min_length=2
    )


class DefaultWorkflow(BaseWorkflow):
    name: ClassVar[str] = "default"
    description: ClassVar[str] = (
        "Suited for tasks that might require thought, planning or subdivision."
    )
    examples: ClassVar[list[str]] = [
        "Writing technical documentation for a software project, because it might involves research and structuring information but not actual software development.",
        "Preparing a report on sales performance, because it might involve data collection, analysis, KPI definition, and selecting a reporting solution.",
    ]


class SoftwareDevelopmentWorkflow(BaseWorkflow):
    name: ClassVar[str] = "software_development"
    description: ClassVar[str] = (
        "For tasks related to software creation or modification, whether in part or whole."
    )
    examples: ClassVar[list[str]] = [
        "Developing a new feature for an existing mobile app.",
        "Updating and fixing bugs in a legacy software system.",
        "Integrating a third-party API into an existing web application.",
    ]


class FastTrackedWorkflow(BaseWorkflow):
    name : ClassVar[str] =  "fast_tracked"
    description: ClassVar[str] = (
        "Ideal for simple tasks that do not require subdivision."
    )
    examples: ClassVar[list[str]] = [
        "Writing 'Hello World' into HelloWorld.txt.",
        "Updating a single value in a database.",
        "Sending a predefined email to a defined list of recipients.",
    ]


class WorkflowRegistry(BaseModel):
    # NOTE: Put the simple/obvious workflows first for LLM understanding
    workflows: dict[str, BaseWorkflow] = {
        FastTrackedWorkflow.name: FastTrackedWorkflow(),
        SoftwareDevelopmentWorkflow.name: SoftwareDevelopmentWorkflow(),
        DefaultWorkflow.name: DefaultWorkflow(),
    }

    def get_workflow(self, name: str) -> BaseWorkflow:
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow
        raise Exception(f"Workflow {name} not found in registry")

    def __str__(self) -> str:
        return str(self.workflows)

    def __repr__(self) -> str:
        return str(self.workflows)

    def __iter__(self):
        return iter(self.workflows)

    def __len__(self):
        return len(self.workflows)
