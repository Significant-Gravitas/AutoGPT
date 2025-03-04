from typing import Any, Dict, Optional
from pydantic import BaseModel


class Document(BaseModel):
    url: str
    relevance_score: float


class ApiResponse(BaseModel):
    answer: str
    documents: list[Document]
    success: bool


class GraphData(BaseModel):
    nodes: list[Dict[str, Any]]
    edges: list[Dict[str, Any]]


class Message(BaseModel):
    query: str
    response: str


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[Message]
    message_id: str
    include_graph_data: bool = False
    graph_id: Optional[str] = None
