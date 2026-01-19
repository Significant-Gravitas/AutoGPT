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
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional

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
DECOMPOSE_SYSTEM_PROMPT = """You are a task decomposition specialist. Your job is to break down a task into actionable sub-steps.

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
{{"sub_items": [{{"content": "Do X", "active_form": "Doing X"}}, {{"content": "Do Y", "active_form": "Doing Y"}}], "summary": "Brief explanation of the breakdown"}}"""


class TodoItem(BaseModel):
    """A single todo item with optional nested sub-items."""

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
        yield "Use todo_write when working on multi-step tasks to track progress"
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

        yield ChatMessage.system("\n".join(lines))

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

        return (
            TodoItem(
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
        result = {
            "content": item.content,
            "status": item.status,
            "active_form": item.active_form,
        }
        if item.sub_items:
            result["sub_items"] = [
                self._serialize_todo_item(sub) for sub in item.sub_items
            ]
        return result

    # -------------------------------------------------------------------------
    # CommandProvider Implementation
    # -------------------------------------------------------------------------

    def get_commands(self) -> Iterator[Command]:
        yield self.todo_write
        yield self.todo_read
        yield self.todo_clear
        yield self.todo_decompose

    @command(
        names=["todo_write"],
        parameters={
            "todos": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description=(
                    "The complete todo list. Each item must have: "
                    "'content' (imperative form like 'Fix bug'), "
                    "'status' (pending|in_progress|completed), "
                    "'active_form' (present continuous like 'Fixing bug'). "
                    "Optional: 'sub_items' (array of nested todo items)"
                ),
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "content": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Imperative form of the task",
                            required=True,
                        ),
                        "status": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Task status: pending, in_progress, or completed",
                            enum=["pending", "in_progress", "completed"],
                            required=True,
                        ),
                        "active_form": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="Present continuous form (shown when in_progress)",
                            required=True,
                        ),
                        "sub_items": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            description="Optional nested sub-tasks (recursive structure)",
                            required=False,
                        ),
                    },
                ),
                required=True,
            ),
        },
    )
    def todo_write(self, todos: list[dict]) -> dict:
        """
        Replace the entire todo list with a new list.

        This is the primary command for managing todos. Use it to:
        - Create initial todos when starting a multi-step task
        - Mark tasks as in_progress when you start working on them
        - Mark tasks as completed when done
        - Add new tasks discovered during work
        - Remove tasks that are no longer relevant
        - Update sub-items created by todo_decompose

        The entire list is replaced atomically, ensuring consistency.
        Supports nested sub_items for hierarchical task tracking.
        """
        # Validate item count
        if len(todos) > self.config.max_items:
            return {
                "status": "error",
                "message": f"Too many items. Maximum is {self.config.max_items}.",
            }

        # Validate and convert items recursively
        validated_items = []
        for i, item in enumerate(todos):
            parsed, error = self._parse_todo_item(item, f"Item {i}")
            if error:
                return {
                    "status": "error",
                    "message": error,
                }
            if parsed:
                validated_items.append(parsed)

        # Count in_progress items and warn if more than one
        in_progress_count = sum(1 for t in validated_items if t.status == "in_progress")
        warning = None
        if in_progress_count > 1:
            warning = (
                f"Warning: {in_progress_count} tasks are in_progress. "
                "Best practice is to have only ONE task in_progress at a time."
            )
            logger.warning(warning)

        # Replace the list
        self._todos = TodoList(items=validated_items)

        # Build response
        pending = sum(1 for t in validated_items if t.status == "pending")
        completed = sum(1 for t in validated_items if t.status == "completed")

        response = {
            "status": "success",
            "item_count": len(validated_items),
            "pending": pending,
            "in_progress": in_progress_count,
            "completed": completed,
        }

        if warning:
            response["warning"] = warning

        return response

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
        if item_index < 0 or item_index >= len(self._todos.items):
            return {
                "status": "error",
                "message": f"Invalid item_index {item_index}. Valid range: 0-{len(self._todos.items) - 1}",
            }

        target_item = self._todos.items[item_index]

        # Check if already has sub-items
        if target_item.sub_items:
            return {
                "status": "error",
                "message": f"Item '{target_item.content}' already has {len(target_item.sub_items)} sub-items. Clear them first if you want to re-decompose.",
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
            response = await self._llm_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(prompt_content)],
                model_name=self.config.decompose_model or self._smart_llm,
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
