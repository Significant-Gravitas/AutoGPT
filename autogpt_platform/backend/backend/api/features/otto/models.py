from typing import Any

from pydantic import BaseModel


class Document(BaseModel):
    url: str
    relevance_score: float


class ApiResponse(BaseModel):
    answer: str
    documents: list[Document]
    success: bool


class GraphData(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    graph_name: str | None = None
    graph_description: str | None = None


class Message(BaseModel):
    query: str
    response: str


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[Message]
    message_id: str
    include_graph_data: bool = False
    graph_id: str | None = None
