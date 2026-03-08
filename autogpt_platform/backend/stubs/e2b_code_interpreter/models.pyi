"""Partial type stub for e2b_code_interpreter.models."""

from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class OutputMessage:
    line: str
    timestamp: int
    error: bool

@dataclass
class Logs:
    stdout: List[str]
    stderr: List[str]

@dataclass
class Result:
    formats: Any
    def text(self) -> Optional[str]: ...

@dataclass
class ExecutionError:
    name: str
    value: str
    traceback_raw: List[str]

@dataclass
class Execution:
    results: List[Result]
    logs: Logs
    error: Optional[ExecutionError]
    text: str

@dataclass
class Context:
    id: str
    language: str
    cwd: Optional[str]
