from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from pydantic import ConfigDict, Field

from AFAAS.configs.schema import AFAASModel
from AFAAS.lib.sdk.artifacts import Artifact

if TYPE_CHECKING:
    from AFAAS.core.agents.planner.main import PlannerAgent


class ArtifactUpload(AFAASModel):
    file: str = Field(..., description="File to upload.", format="binary")
    relative_path: str = Field(
        ...,
        description="Relative path of the artifact in the agent's workspace.",
        examples=["python/code"],
    )


class Pagination(AFAASModel):
    total_items: int = Field(..., description="Total number of items.", examples=[42])
    total_pages: int = Field(..., description="Total number of pages.", examples=[97])
    current_page: int = Field(..., description="Current_page page number.", examples=[1])
    page_size: int = Field(..., description="Number of items per page.", examples=[25])

    @classmethod
    def create_pagination(
        cls, total_items: int, current_page: int, page_size: int
    ) -> Pagination:
        import math

        return cls(
            total_items=total_items,
            current_page=current_page,
            page_size=page_size,
            total_pages=math.ceil(total_items / page_size),
        )


class TaskOutput(AFAASModel):
    pass


class TaskInput(AFAASModel):
    pass


class AgentRequestBody(AFAASModel):
    input: str = Field(
        ...,
        min_length=1,
        description="Input prompt for the task.",
        examples=["Write the words you receive to the file 'output.txt'."],
    )

    # NOTE: Base Agent not planned agent because Pydantic (1.10 (at least)) can't serialize :
    # PlannerAgent
    # chat_model_provider: OpenAISettings =OpenAISettings()
    # tool_registry: DefaultToolRegistry.SystemSettings = DefaultToolRegistry.SystemSettings()
    # prompt_manager: PromptManager.SystemSettings = PromptManager.SystemSettings()

    # additional_input: = Any ?
    # additional_input: Optional[BaseAgent.SystemSettings]

    def json(self, *args, **kwargs):
        return super().json(*args, **kwargs)


class Agent(AgentRequestBody):
    # additional_input: Optional[BaseAgent.SystemSettings]
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    artifacts: Optional[List[Artifact]] = Field(
        [],
        description="A list of artifacts that the task has produced.",
        examples=[[
            "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
            "ab7b4091-2560-4692-a4fe-d831ea3ca7d6",
        ]],
    )
    
    
    model_config = ConfigDict(extra="allow", use_enum_values=True, json_encoders={
        uuid.UUID: lambda v: str(v),
        float: lambda v: str(
            9999.99 if v == float("inf") or v == float("-inf") else v
        ),
        datetime: lambda v: v.isoformat(),
    })

    @classmethod
    def from_afaas(cls, agent: PlannerAgent.SystemSettings):
        return cls(
            task_id=agent.agent_id,
            input=agent.agent_goal_sentence,
            additional_input=agent,
            created_at=agent.created_at,
            modified_at=agent.modified_at,
            # TODO: @lennart02 https://github.com/ph-ausseil/afaas/issues/20
            artifacts=[],  # list_artifacts(agent= agent , agent_id= agent.agent_id )
        )


class AgentListResponse(AFAASModel):
    tasks: Optional[List[Agent]] = None
    pagination: Optional[Pagination] = None

    @classmethod
    def from_afaas(cls, agent_list: list[PlannerAgent.SystemSettings]):
        tasks = []
        for agent in agent_list:
            tasks.append(Agent.from_afaas(agent))
        return cls(tasks=tasks)
    
    
    model_config = ConfigDict(extra="allow", use_enum_values=True, json_encoders={
        uuid.UUID: lambda v: str(v),
        float: lambda v: str(
            9999.99 if v == float("inf") or v == float("-inf") else v
        ),
        datetime: lambda v: v.isoformat(),
    }, allow_inf_nan=False)

    def json(self, *args, **kwargs):
        return super().json(*args, **kwargs)


class TaskRequestBody(AFAASModel):
    name: Optional[str] = Field(
        None, description="The name of the task step.", examples=["Write to file"]
    )
    input: Optional[str] = Field(
        None,
        description="Input prompt for the step.",
        examples=["Washington"],
    )
    additional_input: Optional[dict] = {}


class Status(Enum):
    created = "created"
    running = "running"
    completed = "completed"


class Task(TaskRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    step_id: str = Field(
        ...,
        description="The ID of the task step.",
        examples=["6bb1801a-fd80-45e8-899a-4dd723cc602e"],
    )
    name: Optional[str] = Field(
        None, description="The name of the task step.", examples=["Write to file"]
    )
    status: Status = Field(
        ..., description="The status of the task step.", examples=["created"]
    )
    output: Optional[str] = Field(
        None,
        description="Output of the task step.",
        examples=["I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')"],
    )
    additional_output: Optional[TaskOutput] = {}
    artifacts: Optional[List[Artifact]] = Field(
        [], description="A list of artifacts that the step has produced."
    )
    is_last: bool = Field(
        ..., description="Whether this is the last step in the task.", examples=[True]
    )


class AgentTasksListResponse(AFAASModel):
    steps: Optional[List[Task]] = None
    pagination: Optional[Pagination] = None


class AgentArtifactsListResponse(AFAASModel):
    artifacts: Optional[List[Artifact]] = None
    pagination: Optional[Pagination] = None


class TaskEvalRequestBody(TaskRequestBody):
    eval_id: str
