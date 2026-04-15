"""
Todo Component - Task management for autonomous agents.

A simple, effective task management system modeled after Claude Code's TodoWrite tool.
Agents use this to track multi-step tasks naturally and frequently.

Features:
- Hierarchical task structure with sub-items
- Smart LLM-based task decomposition
- Status tracking at all levels
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider, MessageProvider
from forge.command import Command, command
from forge.llm.providers import ChatMessage
from forge.models.json_schema import JSONSchema

if TYPE_CHECKING:
    from forge.llm.providers import MultiProvider

logger = logging.getLogger(__name__)


# Status type
TodoStatus = Literal["pending", "in_progress", "completed"]

# System prompt for task decomposition
DECOMPOSE_SYSTEM_PROMPT = """\
You are a task decomposition specialist. Break down tasks into actionable sub-steps.

Current Plan Context:
{current_todos}

Task to Decompose:
{task_content}

Additional Context:
{context}

Instructions:
1. Analyze the task and break it into 3-7 concrete sub-steps
2. Each sub-step should be actionable and specific
3. Sub-steps should be in logical order
4. Keep sub-steps concise (1 line each)
5. Generate both imperative (content) and present continuous (active_form) versions

Respond with ONLY a JSON object (no markdown, no explanation):
{{"sub_items": [{{"content": "Do X", "active_form": "Doing X"}}], \
"summary": "Brief explanation"}}"""


def _generate_todo_id() -> str:
    """Generate a short unique ID for todo items."""
    return uuid.uuid4().hex[:8]


class TodoItem(BaseModel):
    """A single todo item with optional nested sub-items."""

    id: str = Field(default_factory=_generate_todo_id, description="Unique identifier")
    content: str = Field(..., description="Imperative form: 'Fix the bug'")
    status: TodoStatus = Field(default="pending", description="Task status")
    active_form: str = Field(
        ..., description="Present continuous form: 'Fixing the bug'"
    )
    sub_items: list["TodoItem"] = Field(
        default_factory=list, description="Nested sub-tasks"
    )

    model_config = ConfigDict(frozen=False)


# Rebuild model to resolve forward reference
TodoItem.model_rebuild()


class TodoList(BaseModel):
    """The complete todo list."""

    items: list[TodoItem] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False)


class TodoConfiguration(BaseModel):
    """Configuration for the Todo component."""

    max_items: int = Field(default=50, description="Maximum number of todos")
    show_in_prompt: bool = Field(
        default=True, description="Whether to include todos in LLM context"
    )
    decompose_model: Optional[str] = Field(
        default=None, description="Model for decomposition (defaults to smart_llm)"
    )

    model_config = ConfigDict(frozen=False)


class TodoComponent(
    DirectiveProvider,
    CommandProvider,
    MessageProvider,
    ConfigurableComponent[TodoConfiguration],
):
    """
    Task management component for tracking multi-step tasks.

    Features:
    - Hierarchical todo list with sub-items
    - Atomic updates (replace entire list)
    - Three statuses: pending, in_progress, completed
    - Dual descriptions (imperative + active form)
    - Smart LLM-based task decomposition
    - Visible in LLM context for awareness
    """

    config_class = TodoConfiguration

    def __init__(
        self,
        llm_provider: Optional["MultiProvider"] = None,
        smart_llm: Optional[str] = None,
        config: Optional[TodoConfiguration] = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self._todos = TodoList()
        self._llm_provider = llm_provider
        self._smart_llm = smart_llm

    # -------------------------------------------------------------------------
    # DirectiveProvider Implementation
    # -------------------------------------------------------------------------

    def get_resources(self) -> Iterator[str]:
        yield "A todo list to track and manage multi-step tasks. Use frequently!"

    def get_best_practices(self) -> Iterator[str]:
        yield "Use todo_bulk_add for initial planning, then incremental ops for updates"
        yield "Use todo_set_status to mark tasks in_progress or completed"
        yield "Use todo_add to add a single new task to an existing list"
        yield "Mark todos as in_progress before starting work on them"
        yield "Mark todos as completed immediately after finishing, not in batches"
        yield "Only have ONE todo as in_progress at a time"

    # -------------------------------------------------------------------------
    # MessageProvider Implementation
    # -------------------------------------------------------------------------

    def _format_todo_item(self, item: TodoItem, indent: int = 0) -> list[str]:
        """Format a todo item with its sub-items recursively."""
        lines = []
        prefix = "  " * indent

        if item.status == "completed":
            lines.append(f"{prefix}- [x] {item.content}")
        elif item.status == "in_progress":
            lines.append(f"{prefix}- [~] {item.active_form}")
        else:
            lines.append(f"{prefix}- [ ] {item.content}")

        # Recursively format sub-items
        for sub in item.sub_items:
            lines.extend(self._format_todo_item(sub, indent + 1))

        return lines

    def _get_current_todos_text(self) -> str:
        """Get a text representation of current todos for the decomposition prompt."""
        if not self._todos.items:
            return "No current todos."

        lines = []
        for i, item in enumerate(self._todos.items):
            lines.extend(self._format_todo_item(item))
        return "\n".join(lines)

    def get_messages(self) -> Iterator[ChatMessage]:
        if not self.config.show_in_prompt or not self._todos.items:
            return

        in_progress = [t for t in self._todos.items if t.status == "in_progress"]
        pending = [t for t in self._todos.items if t.status == "pending"]
        completed = [t for t in self._todos.items if t.status == "completed"]

        lines = ["## Your Todo List\n"]

        # Show in-progress first (most important) with sub-items
        if in_progress:
            lines.append("**Currently working on:**")
            for todo in in_progress:
                lines.extend(self._format_todo_item(todo))

        # Show pending with sub-items
        if pending:
            lines.append("\n**Pending:**")
            for todo in pending:
                lines.extend(self._format_todo_item(todo))

        # Show completed (brief summary)
        if completed:
            lines.append(f"\n**Completed:** {len(completed)} task(s)")

        yield ChatMessage.user("\n".join(lines))

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _parse_todo_item(
        self, item: dict, path: str = "Item"
    ) -> tuple[Optional[TodoItem], Optional[str]]:
        """
        Recursively parse a dict into a TodoItem with sub_items.

        Returns (TodoItem, None) on success or (None, error_message) on failure.
        """
        # Check required fields
        if not item.get("content"):
            return None, f"{path}: 'content' is required and must be non-empty"
        if not item.get("active_form"):
            return None, f"{path}: 'active_form' is required and must be non-empty"
        if item.get("status") not in ("pending", "in_progress", "completed"):
            return (
                None,
                f"{path}: 'status' must be one of: pending, in_progress, completed",
            )

        # Parse sub_items recursively
        sub_items = []
        raw_sub_items = item.get("sub_items", [])
        if raw_sub_items:
            for j, sub_item in enumerate(raw_sub_items):
                parsed, error = self._parse_todo_item(
                    sub_item, f"{path}.sub_items[{j}]"
                )
                if error:
                    return None, error
                if parsed:
                    sub_items.append(parsed)

        # Use provided ID or generate a new one
        item_id = item.get("id") or _generate_todo_id()

        return (
            TodoItem(
                id=item_id,
                content=item["content"],
                status=item["status"],
                active_form=item["active_form"],
                sub_items=sub_items,
            ),
            None,
        )

    def _serialize_todo_item(self, item: TodoItem) -> dict:
        """
        Recursively serialize a TodoItem to a dict including sub_items.
        """
        result: dict[str, str | list] = {
            "id": item.id,
            "content": item.content,
            "status": item.status,
            "active_form": item.active_form,
        }
        if item.sub_items:
            result["sub_items"] = [
                self._serialize_todo_item(sub) for sub in item.sub_items
            ]
        return result

    def _find_by_id(self, todo_id: str) -> Optional[TodoItem]:
        """Find a todo item by its ID (top-level only)."""
        for item in self._todos.items:
            if item.id == todo_id:
                return item
        return None

    def _find_index_by_id(self, todo_id: str) -> int:
        """Find the index of a todo item by its ID. Returns -1 if not found."""
        for i, item in enumerate(self._todos.items):
            if item.id == todo_id:
                return i
        return -1

    # -------------------------------------------------------------------------
    # CommandProvider Implementation
    # -------------------------------------------------------------------------

    def get_commands(self) -> Iterator[Command]:
        # Incremental operations (token efficient)
        yield self.todo_add
        yield self.todo_set_status
        yield self.todo_update
        yield self.todo_delete
        yield self.todo_bulk_add
        yield self.todo_reorder
        # Core operations
        yield self.todo_read
        yield self.todo_clear
        yield self.todo_decompose

    @command(names=["todo_read"])
    def todo_read(self) -> dict:
        """
        Get the current todo list.

        Returns all todos with their current statuses and sub-items.
        Useful for reviewing progress or understanding current state.
        """
        return {
            "status": "success",
            "items": [self._serialize_todo_item(t) for t in self._todos.items],
            "summary": {
                "pending": sum(1 for t in self._todos.items if t.status == "pending"),
                "in_progress": sum(
                    1 for t in self._todos.items if t.status == "in_progress"
                ),
                "completed": sum(
                    1 for t in self._todos.items if t.status == "completed"
                ),
            },
        }

    @command(names=["todo_clear"])
    def todo_clear(self) -> dict:
        """
        Clear all todos.

        Removes all items from the todo list.
        Use when starting fresh or when the current task list is no longer relevant.
        """
        count = len(self._todos.items)
        self._todos = TodoList()

        return {
            "status": "success",
            "message": f"Cleared {count} todo(s)",
        }

    @command(
        names=["todo_decompose"],
        parameters={
            "item_index": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Index of the todo item to decompose (0-based)",
                required=True,
            ),
            "context": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Additional context to help guide the decomposition",
                required=False,
            ),
        },
    )
    async def todo_decompose(self, item_index: int, context: str = "") -> dict:
        """
        Use the smart LLM to break down a todo item into actionable sub-steps.

        This spawns a focused decomposition call with the current plan context.
        The LLM analyzes the task and generates 3-7 concrete sub-steps.

        Requires an LLM provider to be configured for this component.
        """
        # Validate LLM availability
        if not self._llm_provider or not self._smart_llm:
            return {
                "status": "error",
                "message": "LLM provider not configured. Cannot decompose tasks.",
            }

        # Validate item index
        max_idx = len(self._todos.items) - 1
        if item_index < 0 or item_index > max_idx:
            return {
                "status": "error",
                "message": f"Invalid item_index {item_index}. Valid: 0-{max_idx}",
            }

        target_item = self._todos.items[item_index]

        # Check if already has sub-items
        if target_item.sub_items:
            count = len(target_item.sub_items)
            return {
                "status": "error",
                "message": (
                    f"Item '{target_item.content}' already has {count} sub-items. "
                    "Clear them first to re-decompose."
                ),
            }

        # Build the decomposition prompt
        prompt_content = DECOMPOSE_SYSTEM_PROMPT.format(
            current_todos=self._get_current_todos_text(),
            task_content=target_item.content,
            context=context or "No additional context provided.",
        )

        try:
            from forge.llm.providers import ChatMessage

            # Call the LLM for decomposition
            model = self.config.decompose_model or self._smart_llm
            response = await self._llm_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(prompt_content)],
                model_name=model,  # type: ignore[arg-type]
            )

            # Parse the JSON response
            response_text = response.response.content
            if not response_text:
                return {
                    "status": "error",
                    "message": "LLM returned empty response",
                }

            # Try to extract JSON from response (handle potential markdown wrapping)
            json_text = response_text.strip()
            if json_text.startswith("```"):
                # Remove markdown code blocks
                lines = json_text.split("\n")
                json_lines = []
                in_code = False
                for line in lines:
                    if line.startswith("```"):
                        in_code = not in_code
                        continue
                    if in_code or not line.startswith("```"):
                        json_lines.append(line)
                json_text = "\n".join(json_lines)

            decomposition = json.loads(json_text)

            # Validate response structure
            if "sub_items" not in decomposition:
                return {
                    "status": "error",
                    "message": "LLM response missing 'sub_items' field",
                }

            # Create sub-items
            new_sub_items = []
            for sub in decomposition["sub_items"]:
                if not sub.get("content") or not sub.get("active_form"):
                    continue
                new_sub_items.append(
                    TodoItem(
                        content=sub["content"],
                        active_form=sub["active_form"],
                        status="pending",
                    )
                )

            if not new_sub_items:
                return {
                    "status": "error",
                    "message": "LLM generated no valid sub-items",
                }

            # Update the target item with sub-items
            target_item.sub_items = new_sub_items

            return {
                "status": "success",
                "item": target_item.content,
                "sub_items_count": len(new_sub_items),
                "sub_items": [
                    {"content": s.content, "active_form": s.active_form}
                    for s in new_sub_items
                ],
                "summary": decomposition.get("summary", "Task decomposed successfully"),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM decomposition response: {e}")
            return {
                "status": "error",
                "message": f"Failed to parse LLM response as JSON: {e}",
            }
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {
                "status": "error",
                "message": f"Decomposition failed: {e}",
            }

    # -------------------------------------------------------------------------
    # Incremental Operations - Token-efficient todo management
    # -------------------------------------------------------------------------

    @command(
        names=["todo_add"],
        parameters={
            "content": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Imperative form of the task (e.g., 'Fix the bug')",
                required=True,
            ),
            "active_form": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Present continuous form (e.g., 'Fixing the bug')",
                required=True,
            ),
            "status": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Initial status: pending, in_progress, or completed",
                enum=["pending", "in_progress", "completed"],
                required=False,
            ),
            "index": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Position to insert at (0-based). Appends if omitted.",
                required=False,
            ),
        },
    )
    def todo_add(
        self,
        content: str,
        active_form: str,
        status: TodoStatus = "pending",
        index: Optional[int] = None,
    ) -> dict:
        """
        Add a single todo item. Returns the created item with its ID.

        This is the most token-efficient way to add a new task.
        Use this instead of todo_write when adding one item to an existing list.
        """
        # Validate inputs
        if not content or not content.strip():
            return {"status": "error", "message": "'content' is required"}
        if not active_form or not active_form.strip():
            return {"status": "error", "message": "'active_form' is required"}

        # Check max items
        if len(self._todos.items) >= self.config.max_items:
            return {
                "status": "error",
                "message": f"Cannot add: max items ({self.config.max_items}) reached",
            }

        # Create the new item
        new_item = TodoItem(
            content=content.strip(),
            active_form=active_form.strip(),
            status=status,
        )

        # Insert at specified index or append
        if index is not None:
            if index < 0:
                index = 0
            if index > len(self._todos.items):
                index = len(self._todos.items)
            self._todos.items.insert(index, new_item)
        else:
            self._todos.items.append(new_item)

        return {
            "status": "success",
            "item": self._serialize_todo_item(new_item),
            "total_items": len(self._todos.items),
        }

    @command(
        names=["todo_set_status"],
        parameters={
            "id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The unique ID of the todo to update",
                required=True,
            ),
            "status": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="New status: pending, in_progress, or completed",
                enum=["pending", "in_progress", "completed"],
                required=True,
            ),
        },
    )
    def todo_set_status(self, id: str, status: TodoStatus) -> dict:
        """
        Update just the status of a todo by ID.

        This is the most common operation and the most token-efficient way
        to mark a task as in_progress or completed.
        """
        item = self._find_by_id(id)
        if not item:
            return {"status": "error", "message": f"Todo with ID '{id}' not found"}

        old_status = item.status
        item.status = status

        return {
            "status": "success",
            "item": self._serialize_todo_item(item),
            "changed": {"status": {"from": old_status, "to": status}},
        }

    @command(
        names=["todo_update"],
        parameters={
            "id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The unique ID of the todo to update",
                required=True,
            ),
            "content": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="New imperative form (optional)",
                required=False,
            ),
            "active_form": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="New present continuous form (optional)",
                required=False,
            ),
            "status": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="New status (optional)",
                enum=["pending", "in_progress", "completed"],
                required=False,
            ),
        },
    )
    def todo_update(
        self,
        id: str,
        content: Optional[str] = None,
        active_form: Optional[str] = None,
        status: Optional[TodoStatus] = None,
    ) -> dict:
        """
        Partial update of a todo - only specified fields change.

        Use this when you need to update multiple fields at once.
        For just status changes, prefer todo_set_status.
        """
        item = self._find_by_id(id)
        if not item:
            return {"status": "error", "message": f"Todo with ID '{id}' not found"}

        changes: dict[str, dict[str, str]] = {}

        if content is not None:
            if not content.strip():
                return {"status": "error", "message": "'content' cannot be empty"}
            changes["content"] = {"from": item.content, "to": content.strip()}
            item.content = content.strip()

        if active_form is not None:
            if not active_form.strip():
                return {"status": "error", "message": "'active_form' cannot be empty"}
            changes["active_form"] = {
                "from": item.active_form,
                "to": active_form.strip(),
            }
            item.active_form = active_form.strip()

        if status is not None:
            changes["status"] = {"from": item.status, "to": status}
            item.status = status

        if not changes:
            return {
                "status": "success",
                "item": self._serialize_todo_item(item),
                "message": "No changes specified",
            }

        return {
            "status": "success",
            "item": self._serialize_todo_item(item),
            "changed": changes,
        }

    @command(
        names=["todo_delete"],
        parameters={
            "id": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The unique ID of the todo to delete",
                required=True,
            ),
        },
    )
    def todo_delete(self, id: str) -> dict:
        """
        Explicitly delete a todo by ID.

        Unlike todo_write where items are removed by omission (easy to accidentally
        delete), this is an explicit delete operation.
        """
        index = self._find_index_by_id(id)
        if index == -1:
            return {"status": "error", "message": f"Todo with ID '{id}' not found"}

        deleted_item = self._todos.items.pop(index)

        return {
            "status": "success",
            "deleted": self._serialize_todo_item(deleted_item),
            "remaining_items": len(self._todos.items),
        }

    @command(
        names=["todo_bulk_add"],
        parameters={
            "items": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description="Array of todo items to add",
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "content": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Imperative form of the task",
                            required=True,
                        ),
                        "active_form": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Present continuous form",
                            required=True,
                        ),
                        "status": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Initial status (default: pending)",
                            enum=["pending", "in_progress", "completed"],
                            required=False,
                        ),
                    },
                ),
                required=True,
            ),
        },
    )
    def todo_bulk_add(self, items: list[dict]) -> dict:
        """
        Add multiple todos at once. Use for initial planning.

        This is efficient for creating the initial todo list at the start
        of a multi-step task. For subsequent additions, use todo_add.
        """
        if not items:
            return {"status": "error", "message": "No items provided"}

        # Check max items
        if len(self._todos.items) + len(items) > self.config.max_items:
            return {
                "status": "error",
                "message": (
                    f"Cannot add {len(items)} items: would exceed max "
                    f"({self.config.max_items}). Current: {len(self._todos.items)}"
                ),
            }

        added_items = []
        for i, item in enumerate(items):
            content = item.get("content", "").strip()
            active_form = item.get("active_form", "").strip()
            status = item.get("status", "pending")

            if not content:
                return {
                    "status": "error",
                    "message": f"Item {i}: 'content' is required",
                }
            if not active_form:
                return {
                    "status": "error",
                    "message": f"Item {i}: 'active_form' is required",
                }
            if status not in ("pending", "in_progress", "completed"):
                return {
                    "status": "error",
                    "message": f"Item {i}: invalid status '{status}'",
                }

            new_item = TodoItem(
                content=content,
                active_form=active_form,
                status=status,
            )
            self._todos.items.append(new_item)
            added_items.append(self._serialize_todo_item(new_item))

        return {
            "status": "success",
            "added": added_items,
            "added_count": len(added_items),
            "total_items": len(self._todos.items),
        }

    @command(
        names=["todo_reorder"],
        parameters={
            "ids": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description="List of todo IDs in the desired order",
                items=JSONSchema(type=JSONSchema.Type.STRING),
                required=True,
            ),
        },
    )
    def todo_reorder(self, ids: list[str]) -> dict:
        """
        Reorder todos by providing the ID list in desired order.

        All current todo IDs must be included. This operation only
        changes the order, not the items themselves.
        """
        current_ids = {item.id for item in self._todos.items}
        provided_ids = set(ids)

        # Check for duplicates first (before other checks)
        if len(ids) != len(provided_ids):
            return {"status": "error", "message": "Duplicate IDs in reorder list"}

        # Validate that all provided IDs exist
        unknown = provided_ids - current_ids
        if unknown:
            return {
                "status": "error",
                "message": f"Unknown todo IDs: {', '.join(unknown)}",
            }

        # Validate that all current IDs are provided
        missing = current_ids - provided_ids
        if missing:
            return {
                "status": "error",
                "message": f"Missing todo IDs in reorder list: {', '.join(missing)}",
            }

        # Reorder
        id_to_item = {item.id: item for item in self._todos.items}
        self._todos.items = [id_to_item[id] for id in ids]

        return {
            "status": "success",
            "order": ids,
            "message": f"Reordered {len(ids)} items",
        }
