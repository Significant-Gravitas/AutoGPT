"""
Task Queue Block — manages a persistent queue of coding tasks for sequential execution.

Tasks are stored in a JSON file (local) and processed one at a time.
Supports: enqueue, dequeue, peek, list, clear, and status operations.
Integrates with the agent loop to automatically pick up the next task.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_FILE = "./data/task_queue.json"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueueOperation(str, Enum):
    ENQUEUE = "enqueue"
    DEQUEUE = "dequeue"
    PEEK = "peek"
    LIST = "list"
    CLEAR = "clear"
    STATUS = "status"
    COMPLETE = "complete"
    FAIL = "fail"


def _load_queue(queue_file: str) -> list[dict]:
    path = Path(queue_file)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _save_queue(queue_file: str, queue: list[dict]) -> None:
    path = Path(queue_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2, default=str))


class TaskQueueInput(BlockSchemaInput):
    operation: QueueOperation = SchemaField(
        default=QueueOperation.ENQUEUE,
        description="Queue operation: enqueue, dequeue, peek, list, clear, status, complete, fail.",
    )
    task_prompt: str = SchemaField(
        default="",
        description="Task description to enqueue (used for ENQUEUE operation).",
    )
    task_id: str = SchemaField(
        default="",
        description="Task ID for COMPLETE, FAIL, or STATUS operations.",
    )
    priority: int = SchemaField(
        default=5,
        description="Task priority (1=highest, 10=lowest). Used for ordering.",
    )
    persona: str = SchemaField(
        default="fullstack_dev",
        description="Agent persona to use for this task.",
    )
    model_mode: str = SchemaField(
        default="auto",
        description="Model mode: 'auto', 'standard', or 'max'.",
    )
    tags: list = SchemaField(
        default_factory=list,
        description="Optional tags for categorizing tasks (e.g., ['frontend', 'bugfix']).",
    )
    queue_file: str = SchemaField(
        default=DEFAULT_QUEUE_FILE,
        description="Path to the JSON file used as the task queue.",
    )
    error_message: str = SchemaField(
        default="",
        description="Error message for FAIL operation.",
    )


class TaskQueueOutput(BlockSchemaOutput):
    task_id: str = SchemaField(description="ID of the affected task.")
    task_prompt: str = SchemaField(description="Task prompt of the dequeued/peeked task.")
    queue_length: int = SchemaField(description="Current number of pending tasks in the queue.")
    status: str = SchemaField(description="Operation result status message.")
    tasks: list = SchemaField(description="List of tasks (for LIST operation).")
    persona: str = SchemaField(description="Persona assigned to the task.")
    model_mode: str = SchemaField(description="Model mode assigned to the task.")


class TaskQueueBlock(Block):
    """
    Persistent task queue for the coding agent.

    Queue multiple tasks to run sequentially. Each task stores its prompt,
    priority, persona, model mode, and status. The agent automatically
    dequeues and processes tasks in priority order.
    """

    class Input(TaskQueueInput):
        pass

    class Output(TaskQueueOutput):
        pass

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-efab-567890123456",
            description=(
                "Persistent task queue for sequential agent execution. "
                "Enqueue multiple coding tasks and process them in priority order."
            ),
            categories={BlockCategory.AI},
            input_schema=TaskQueueBlock.Input,
            output_schema=TaskQueueBlock.Output,
            test_input={
                "operation": QueueOperation.ENQUEUE.value,
                "task_prompt": "Add dark mode support to the settings page.",
                "priority": 3,
                "persona": "frontend_dev",
                "model_mode": "standard",
                "queue_file": "/tmp/test_queue.json",
            },
            test_output=[
                ("status", "Task enqueued successfully."),
                ("queue_length", 1),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        queue = _load_queue(input_data.queue_file)

        if input_data.operation == QueueOperation.ENQUEUE:
            task = {
                "id": str(uuid.uuid4()),
                "prompt": input_data.task_prompt,
                "priority": input_data.priority,
                "persona": input_data.persona,
                "model_mode": input_data.model_mode,
                "tags": input_data.tags,
                "status": TaskStatus.PENDING.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None,
            }
            queue.append(task)
            # Sort by priority (ascending = higher priority first)
            queue.sort(key=lambda t: (t.get("priority", 5), t.get("created_at", "")))
            _save_queue(input_data.queue_file, queue)
            pending = [t for t in queue if t["status"] == TaskStatus.PENDING.value]
            yield "task_id", task["id"]
            yield "task_prompt", task["prompt"]
            yield "queue_length", len(pending)
            yield "status", "Task enqueued successfully."
            yield "tasks", []
            yield "persona", task["persona"]
            yield "model_mode", task["model_mode"]

        elif input_data.operation == QueueOperation.DEQUEUE:
            pending = [t for t in queue if t["status"] == TaskStatus.PENDING.value]
            if not pending:
                yield "task_id", ""
                yield "task_prompt", ""
                yield "queue_length", 0
                yield "status", "Queue is empty."
                yield "tasks", []
                yield "persona", ""
                yield "model_mode", ""
                return
            task = pending[0]
            task["status"] = TaskStatus.RUNNING.value
            task["started_at"] = datetime.now(timezone.utc).isoformat()
            _save_queue(input_data.queue_file, queue)
            remaining = len([t for t in queue if t["status"] == TaskStatus.PENDING.value])
            yield "task_id", task["id"]
            yield "task_prompt", task["prompt"]
            yield "queue_length", remaining
            yield "status", f"Task {task['id']} dequeued and marked as RUNNING."
            yield "tasks", []
            yield "persona", task.get("persona", "fullstack_dev")
            yield "model_mode", task.get("model_mode", "auto")

        elif input_data.operation == QueueOperation.PEEK:
            pending = [t for t in queue if t["status"] == TaskStatus.PENDING.value]
            if not pending:
                yield "task_id", ""
                yield "task_prompt", ""
                yield "queue_length", 0
                yield "status", "Queue is empty."
                yield "tasks", []
                yield "persona", ""
                yield "model_mode", ""
                return
            task = pending[0]
            yield "task_id", task["id"]
            yield "task_prompt", task["prompt"]
            yield "queue_length", len(pending)
            yield "status", "Next task peeked."
            yield "tasks", []
            yield "persona", task.get("persona", "fullstack_dev")
            yield "model_mode", task.get("model_mode", "auto")

        elif input_data.operation == QueueOperation.LIST:
            pending = [t for t in queue if t["status"] == TaskStatus.PENDING.value]
            yield "task_id", ""
            yield "task_prompt", ""
            yield "queue_length", len(pending)
            yield "status", f"{len(queue)} total tasks, {len(pending)} pending."
            yield "tasks", queue
            yield "persona", ""
            yield "model_mode", ""

        elif input_data.operation == QueueOperation.CLEAR:
            queue = []
            _save_queue(input_data.queue_file, queue)
            yield "task_id", ""
            yield "task_prompt", ""
            yield "queue_length", 0
            yield "status", "Queue cleared."
            yield "tasks", []
            yield "persona", ""
            yield "model_mode", ""

        elif input_data.operation in (QueueOperation.COMPLETE, QueueOperation.FAIL):
            task = next((t for t in queue if t["id"] == input_data.task_id), None)
            if not task:
                yield "task_id", input_data.task_id
                yield "task_prompt", ""
                yield "queue_length", len([t for t in queue if t["status"] == TaskStatus.PENDING.value])
                yield "status", f"Task {input_data.task_id} not found."
                yield "tasks", []
                yield "persona", ""
                yield "model_mode", ""
                return
            if input_data.operation == QueueOperation.COMPLETE:
                task["status"] = TaskStatus.COMPLETED.value
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
                msg = f"Task {input_data.task_id} marked as COMPLETED."
            else:
                task["status"] = TaskStatus.FAILED.value
                task["error"] = input_data.error_message
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
                msg = f"Task {input_data.task_id} marked as FAILED."
            _save_queue(input_data.queue_file, queue)
            pending = len([t for t in queue if t["status"] == TaskStatus.PENDING.value])
            yield "task_id", task["id"]
            yield "task_prompt", task["prompt"]
            yield "queue_length", pending
            yield "status", msg
            yield "tasks", []
            yield "persona", task.get("persona", "")
            yield "model_mode", task.get("model_mode", "")

        elif input_data.operation == QueueOperation.STATUS:
            if input_data.task_id:
                task = next((t for t in queue if t["id"] == input_data.task_id), None)
                if task:
                    yield "task_id", task["id"]
                    yield "task_prompt", task["prompt"]
                    yield "queue_length", len([t for t in queue if t["status"] == TaskStatus.PENDING.value])
                    yield "status", task["status"]
                    yield "tasks", [task]
                    yield "persona", task.get("persona", "")
                    yield "model_mode", task.get("model_mode", "")
                else:
                    yield "task_id", input_data.task_id
                    yield "task_prompt", ""
                    yield "queue_length", 0
                    yield "status", "Task not found."
                    yield "tasks", []
                    yield "persona", ""
                    yield "model_mode", ""
            else:
                # Return overall queue stats
                by_status = {}
                for t in queue:
                    s = t.get("status", "unknown")
                    by_status[s] = by_status.get(s, 0) + 1
                pending = by_status.get(TaskStatus.PENDING.value, 0)
                yield "task_id", ""
                yield "task_prompt", ""
                yield "queue_length", pending
                yield "status", json.dumps(by_status)
                yield "tasks", queue
                yield "persona", ""
                yield "model_mode", ""
