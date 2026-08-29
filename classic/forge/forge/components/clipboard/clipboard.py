import json
import logging
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError

logger = logging.getLogger(__name__)


class ClipboardConfiguration(BaseModel):
    max_items: int = Field(
        default=100, description="Maximum number of clipboard items to store"
    )
    max_value_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum size of a single clipboard value in bytes",
    )


class ClipboardComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[ClipboardConfiguration]
):
    """In-memory clipboard for storing and retrieving data between commands."""

    config_class = ClipboardConfiguration

    def __init__(self, config: Optional[ClipboardConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        self._storage: dict[str, Any] = {}

    def get_resources(self) -> Iterator[str]:
        yield "In-memory clipboard for storing temporary data."

    def get_commands(self) -> Iterator[Command]:
        yield self.clipboard_copy
        yield self.clipboard_paste
        yield self.clipboard_list
        yield self.clipboard_clear

    @command(
        ["clipboard_copy", "store", "remember"],
        "Store a value in the clipboard with a key for later retrieval.",
        {
            "key": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A unique key to identify this data",
                required=True,
            ),
            "value": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The value to store (can be any string, including JSON)",
                required=True,
            ),
        },
    )
    def clipboard_copy(self, key: str, value: str) -> str:
        """Store a value in the clipboard.

        Args:
            key: The key to store under
            value: The value to store

        Returns:
            str: Confirmation message
        """
        if not key:
            raise CommandExecutionError("Key cannot be empty")

        # Check value size
        value_size = len(value.encode("utf-8"))
        max_size = self.config.max_value_size
        if value_size > max_size:
            raise CommandExecutionError(
                f"Value too large: {value_size} bytes (max: {max_size})"
            )

        # Check item limit (excluding update of existing key)
        if key not in self._storage and len(self._storage) >= self.config.max_items:
            raise CommandExecutionError(
                f"Clipboard full: max {self.config.max_items} items. "
                "Use clipboard_clear to remove items."
            )

        is_update = key in self._storage
        self._storage[key] = value

        action = "Updated" if is_update else "Stored"
        return json.dumps(
            {
                "action": action.lower(),
                "key": key,
                "value_length": len(value),
                "message": f"{action} value under key '{key}'",
            }
        )

    @command(
        ["clipboard_paste", "retrieve", "recall"],
        "Retrieve a value from the clipboard by its key.",
        {
            "key": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The key of the value to retrieve",
                required=True,
            ),
        },
    )
    def clipboard_paste(self, key: str) -> str:
        """Retrieve a value from the clipboard.

        Args:
            key: The key to retrieve

        Returns:
            str: The stored value or error message
        """
        if key not in self._storage:
            available = list(self._storage.keys())[:10]
            raise CommandExecutionError(
                f"Key '{key}' not found in clipboard. "
                f"Available keys: {available if available else '(empty)'}"
            )

        value = self._storage[key]

        return json.dumps({"key": key, "value": value, "found": True})

    @command(
        ["clipboard_list", "list_stored"],
        "List all keys stored in the clipboard with their value lengths.",
        {},
    )
    def clipboard_list(self) -> str:
        """List all clipboard keys.

        Returns:
            str: JSON with all keys and metadata
        """
        items = []
        for key, value in self._storage.items():
            items.append(
                {
                    "key": key,
                    "value_length": len(str(value)),
                    "value_preview": str(value)[:50]
                    + ("..." if len(str(value)) > 50 else ""),
                }
            )

        return json.dumps(
            {"count": len(items), "items": items, "max_items": self.config.max_items},
            indent=2,
        )

    @command(
        ["clipboard_clear", "forget"],
        "Clear one or all items from the clipboard.",
        {
            "key": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Specific key to clear (omit to clear all)",
                required=False,
            ),
        },
    )
    def clipboard_clear(self, key: str | None = None) -> str:
        """Clear clipboard items.

        Args:
            key: Specific key to clear, or None to clear all

        Returns:
            str: Confirmation message
        """
        if key is not None:
            if key not in self._storage:
                raise CommandExecutionError(f"Key '{key}' not found in clipboard")

            del self._storage[key]
            return json.dumps(
                {"action": "cleared", "key": key, "message": f"Removed key '{key}'"}
            )
        else:
            count = len(self._storage)
            self._storage.clear()
            return json.dumps(
                {
                    "action": "cleared_all",
                    "items_removed": count,
                    "message": f"Cleared {count} item(s) from clipboard",
                }
            )
