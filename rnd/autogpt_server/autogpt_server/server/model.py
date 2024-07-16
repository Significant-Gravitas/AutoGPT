import enum
import typing

import pydantic

import autogpt_server.data.graph


class Methods(enum.Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    EXECUTION_EVENT = "execution_event"
    GET_BLOCKS = "get_blocks"
    EXECUTE_BLOCK = "execute_block"
    GET_GRAPHS = "get_graphs"
    GET_GRAPH = "get_graph"
    CREATE_GRAPH = "create_graph"
    RUN_GRAPH = "run_graph"
    GET_GRAPH_RUNS = "get_graph_runs"
    CREATE_SCHEDULED_RUN = "create_scheduled_run"
    GET_SCHEDULED_RUNS = "get_scheduled_runs"
    UPDATE_SCHEDULED_RUN = "update_scheduled_run"
    UPDATE_CONFIG = "update_config"
    ERROR = "error"


class WsMessage(pydantic.BaseModel):
    method: Methods
    data: typing.Dict[str, typing.Any] | list[typing.Any] | None = None
    success: bool | None = None
    channel: str | None = None
    error: str | None = None


class ExecutionSubscription(pydantic.BaseModel):
    graph_id: str


class SubscriptionDetails(pydantic.BaseModel):
    event_type: str
    channel: str
    graph_id: str


class CreateGraph(pydantic.BaseModel):
    template_id: str | None = None
    template_version: int | None = None
    graph: autogpt_server.data.graph.Graph | None = None


class SetGraphActiveVersion(pydantic.BaseModel):
    active_graph_version: int
