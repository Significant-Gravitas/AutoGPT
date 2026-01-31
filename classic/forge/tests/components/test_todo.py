"""Tests for TodoComponent."""

import pytest

from forge.components.todo import TodoComponent, TodoConfiguration


@pytest.fixture
def todo_component():
    """Create a fresh TodoComponent for testing."""
    return TodoComponent()


class TestTodoRead:
    """Tests for the todo_read command."""

    def test_read_empty_list(self, todo_component):
        """Reading an empty list should return empty items."""
        result = todo_component.todo_read()
        assert result["status"] == "success"
        assert result["items"] == []
        assert result["summary"]["pending"] == 0

    def test_read_after_add(self, todo_component):
        """Reading after adding should return the added items."""
        todo_component.todo_add(
            content="Fix bug", active_form="Fixing bug", status="pending"
        )

        result = todo_component.todo_read()
        assert result["status"] == "success"
        assert len(result["items"]) == 1
        assert result["items"][0]["content"] == "Fix bug"
        assert result["items"][0]["status"] == "pending"
        assert result["items"][0]["active_form"] == "Fixing bug"


class TestTodoClear:
    """Tests for the todo_clear command."""

    def test_clear_empty_list(self, todo_component):
        """Clearing an empty list should succeed."""
        result = todo_component.todo_clear()
        assert result["status"] == "success"
        assert "Cleared 0 todo(s)" in result["message"]

    def test_clear_populated_list(self, todo_component):
        """Clearing a populated list should remove all items."""
        todo_component.todo_bulk_add(
            items=[
                {"content": "Task 1", "active_form": "Task 1"},
                {"content": "Task 2", "active_form": "Task 2"},
            ]
        )

        result = todo_component.todo_clear()
        assert result["status"] == "success"
        assert "Cleared 2 todo(s)" in result["message"]

        # Verify list is empty
        read_result = todo_component.todo_read()
        assert len(read_result["items"]) == 0


class TestProtocols:
    """Tests for protocol implementations."""

    def test_get_resources(self, todo_component):
        """DirectiveProvider.get_resources should yield a resource."""
        resources = list(todo_component.get_resources())
        assert len(resources) == 1
        assert "todo list" in resources[0].lower()

    def test_get_best_practices(self, todo_component):
        """DirectiveProvider.get_best_practices should yield practices."""
        practices = list(todo_component.get_best_practices())
        assert len(practices) == 6
        assert any("todo_bulk_add" in p for p in practices)
        assert any("todo_set_status" in p for p in practices)
        assert any("in_progress" in p for p in practices)

    def test_get_commands(self, todo_component):
        """CommandProvider.get_commands should yield commands."""
        commands = list(todo_component.get_commands())
        command_names = [c.names[0] for c in commands]
        assert "todo_add" in command_names
        assert "todo_read" in command_names
        assert "todo_clear" in command_names

    def test_get_messages_empty_list(self, todo_component):
        """MessageProvider should not yield messages for empty list."""
        messages = list(todo_component.get_messages())
        assert len(messages) == 0

    def test_get_messages_with_todos(self, todo_component):
        """MessageProvider should include todos in LLM context."""
        todo_component.todo_bulk_add(
            items=[
                {
                    "content": "Implement feature",
                    "active_form": "Implementing feature",
                    "status": "in_progress",
                },
                {
                    "content": "Write tests",
                    "active_form": "Writing tests",
                    "status": "pending",
                },
            ]
        )

        messages = list(todo_component.get_messages())
        assert len(messages) == 1

        content = messages[0].content
        assert "Your Todo List" in content
        assert "Currently working on" in content
        assert "Implementing feature" in content
        assert "Pending" in content
        assert "Write tests" in content

    def test_get_messages_respects_show_in_prompt_config(self):
        """MessageProvider should respect show_in_prompt config."""
        component = TodoComponent(config=TodoConfiguration(show_in_prompt=False))
        component.todo_add(content="Task", active_form="Task")

        messages = list(component.get_messages())
        assert len(messages) == 0


class TestConfiguration:
    """Tests for TodoConfiguration."""

    def test_default_configuration(self):
        """Default configuration should have expected values."""
        config = TodoConfiguration()
        assert config.max_items == 50
        assert config.show_in_prompt is True

    def test_custom_configuration(self):
        """Custom configuration should be respected."""
        cfg = TodoConfiguration(max_items=10, show_in_prompt=False)
        component = TodoComponent(config=cfg)
        assert component.config.max_items == 10
        assert component.config.show_in_prompt is False


class TestTodoDecompose:
    """Tests for the todo_decompose command."""

    def test_decompose_without_llm_provider(self, todo_component):
        """Decompose should fail gracefully without LLM provider."""
        todo_component.todo_add(
            content="Complex task", active_form="Complex task", status="pending"
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            todo_component.todo_decompose(item_index=0)
        )
        assert result["status"] == "error"
        assert "LLM provider not configured" in result["message"]

    def test_decompose_empty_list(self, todo_component):
        """Decompose on empty list should fail."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            todo_component.todo_decompose(item_index=0)
        )
        assert result["status"] == "error"

    def test_get_commands_includes_decompose(self, todo_component):
        """CommandProvider should include todo_decompose command."""
        commands = list(todo_component.get_commands())
        command_names = [c.names[0] for c in commands]
        assert "todo_decompose" in command_names


class TestTodoAdd:
    """Tests for the todo_add incremental command."""

    def test_add_single_todo(self, todo_component):
        """Adding a single todo should succeed and return the item with ID."""
        result = todo_component.todo_add(
            content="Fix the bug", active_form="Fixing the bug"
        )
        assert result["status"] == "success"
        assert result["item"]["content"] == "Fix the bug"
        assert result["item"]["active_form"] == "Fixing the bug"
        assert result["item"]["status"] == "pending"
        assert "id" in result["item"]
        assert result["total_items"] == 1

    def test_add_with_status(self, todo_component):
        """Adding a todo with explicit status should work."""
        result = todo_component.todo_add(
            content="Task", active_form="Doing task", status="in_progress"
        )
        assert result["status"] == "success"
        assert result["item"]["status"] == "in_progress"

    def test_add_at_index(self, todo_component):
        """Adding a todo at specific index should insert correctly."""
        # Add two items first
        todo_component.todo_add(content="First", active_form="First")
        todo_component.todo_add(content="Third", active_form="Third")

        # Insert at index 1
        result = todo_component.todo_add(
            content="Second", active_form="Second", index=1
        )
        assert result["status"] == "success"

        # Verify order
        read_result = todo_component.todo_read()
        assert read_result["items"][0]["content"] == "First"
        assert read_result["items"][1]["content"] == "Second"
        assert read_result["items"][2]["content"] == "Third"

    def test_add_validates_empty_content(self, todo_component):
        """Adding with empty content should fail."""
        result = todo_component.todo_add(content="", active_form="Doing something")
        assert result["status"] == "error"
        assert "content" in result["message"]

    def test_add_validates_empty_active_form(self, todo_component):
        """Adding with empty active_form should fail."""
        result = todo_component.todo_add(content="Do something", active_form="")
        assert result["status"] == "error"
        assert "active_form" in result["message"]

    def test_add_enforces_max_items(self):
        """Adding should fail when max items reached."""
        component = TodoComponent(config=TodoConfiguration(max_items=2))
        component.todo_add(content="Task 1", active_form="Task 1")
        component.todo_add(content="Task 2", active_form="Task 2")

        result = component.todo_add(content="Task 3", active_form="Task 3")
        assert result["status"] == "error"
        assert "max items" in result["message"]


class TestTodoSetStatus:
    """Tests for the todo_set_status incremental command."""

    def test_set_status_pending_to_in_progress(self, todo_component):
        """Changing status from pending to in_progress should work."""
        add_result = todo_component.todo_add(content="Task", active_form="Task")
        item_id = add_result["item"]["id"]

        result = todo_component.todo_set_status(id=item_id, status="in_progress")
        assert result["status"] == "success"
        assert result["item"]["status"] == "in_progress"
        assert result["changed"]["status"]["from"] == "pending"
        assert result["changed"]["status"]["to"] == "in_progress"

    def test_set_status_to_completed(self, todo_component):
        """Marking a task as completed should work."""
        add_result = todo_component.todo_add(
            content="Task", active_form="Task", status="in_progress"
        )
        item_id = add_result["item"]["id"]

        result = todo_component.todo_set_status(id=item_id, status="completed")
        assert result["status"] == "success"
        assert result["item"]["status"] == "completed"

    def test_set_status_invalid_id(self, todo_component):
        """Setting status with invalid ID should fail."""
        result = todo_component.todo_set_status(id="nonexistent", status="completed")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestTodoUpdate:
    """Tests for the todo_update incremental command."""

    def test_update_content_only(self, todo_component):
        """Updating only content should preserve other fields."""
        add_result = todo_component.todo_add(
            content="Original", active_form="Original form", status="pending"
        )
        item_id = add_result["item"]["id"]

        result = todo_component.todo_update(id=item_id, content="Updated")
        assert result["status"] == "success"
        assert result["item"]["content"] == "Updated"
        assert result["item"]["active_form"] == "Original form"  # Unchanged
        assert result["item"]["status"] == "pending"  # Unchanged
        assert "content" in result["changed"]
        assert "active_form" not in result["changed"]

    def test_update_multiple_fields(self, todo_component):
        """Updating multiple fields at once should work."""
        add_result = todo_component.todo_add(content="Task", active_form="Task")
        item_id = add_result["item"]["id"]

        result = todo_component.todo_update(
            id=item_id,
            content="New content",
            active_form="New form",
            status="in_progress",
        )
        assert result["status"] == "success"
        assert result["item"]["content"] == "New content"
        assert result["item"]["active_form"] == "New form"
        assert result["item"]["status"] == "in_progress"
        assert len(result["changed"]) == 3

    def test_update_no_changes(self, todo_component):
        """Calling update with no changes should return success with message."""
        add_result = todo_component.todo_add(content="Task", active_form="Task")
        item_id = add_result["item"]["id"]

        result = todo_component.todo_update(id=item_id)
        assert result["status"] == "success"
        assert "No changes" in result["message"]

    def test_update_invalid_id(self, todo_component):
        """Updating with invalid ID should fail."""
        result = todo_component.todo_update(id="nonexistent", content="New")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_update_validates_empty_content(self, todo_component):
        """Updating content to empty should fail."""
        add_result = todo_component.todo_add(content="Task", active_form="Task")
        item_id = add_result["item"]["id"]

        result = todo_component.todo_update(id=item_id, content="")
        assert result["status"] == "error"
        assert "content" in result["message"]


class TestTodoDelete:
    """Tests for the todo_delete incremental command."""

    def test_delete_existing_todo(self, todo_component):
        """Deleting an existing todo should succeed."""
        add_result = todo_component.todo_add(content="Task", active_form="Task")
        item_id = add_result["item"]["id"]

        result = todo_component.todo_delete(id=item_id)
        assert result["status"] == "success"
        assert result["deleted"]["id"] == item_id
        assert result["remaining_items"] == 0

        # Verify it's gone
        read_result = todo_component.todo_read()
        assert len(read_result["items"]) == 0

    def test_delete_from_middle(self, todo_component):
        """Deleting from middle of list should preserve order."""
        todo_component.todo_add(content="First", active_form="First")
        add_result = todo_component.todo_add(content="Second", active_form="Second")
        todo_component.todo_add(content="Third", active_form="Third")

        result = todo_component.todo_delete(id=add_result["item"]["id"])
        assert result["status"] == "success"
        assert result["remaining_items"] == 2

        read_result = todo_component.todo_read()
        assert read_result["items"][0]["content"] == "First"
        assert read_result["items"][1]["content"] == "Third"

    def test_delete_invalid_id(self, todo_component):
        """Deleting with invalid ID should fail."""
        result = todo_component.todo_delete(id="nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestTodoBulkAdd:
    """Tests for the todo_bulk_add command."""

    def test_bulk_add_multiple_items(self, todo_component):
        """Bulk adding multiple items should succeed."""
        result = todo_component.todo_bulk_add(
            items=[
                {"content": "Task 1", "active_form": "Task 1"},
                {"content": "Task 2", "active_form": "Task 2", "status": "in_progress"},
                {"content": "Task 3", "active_form": "Task 3"},
            ]
        )
        assert result["status"] == "success"
        assert result["added_count"] == 3
        assert result["total_items"] == 3
        assert len(result["added"]) == 3

        # Each item should have an ID
        for item in result["added"]:
            assert "id" in item

    def test_bulk_add_empty_list(self, todo_component):
        """Bulk adding empty list should fail."""
        result = todo_component.todo_bulk_add(items=[])
        assert result["status"] == "error"
        assert "No items" in result["message"]

    def test_bulk_add_validates_content(self, todo_component):
        """Bulk add should validate each item's content."""
        result = todo_component.todo_bulk_add(
            items=[
                {"content": "Valid", "active_form": "Valid"},
                {"content": "", "active_form": "Invalid"},
            ]
        )
        assert result["status"] == "error"
        assert "Item 1" in result["message"]
        assert "content" in result["message"]

    def test_bulk_add_validates_active_form(self, todo_component):
        """Bulk add should validate each item's active_form."""
        result = todo_component.todo_bulk_add(
            items=[
                {"content": "Valid", "active_form": ""},
            ]
        )
        assert result["status"] == "error"
        assert "active_form" in result["message"]

    def test_bulk_add_validates_status(self, todo_component):
        """Bulk add should validate each item's status."""
        result = todo_component.todo_bulk_add(
            items=[
                {"content": "Task", "active_form": "Task", "status": "invalid"},
            ]
        )
        assert result["status"] == "error"
        assert "status" in result["message"]

    def test_bulk_add_enforces_max_items(self):
        """Bulk add should respect max items limit."""
        component = TodoComponent(config=TodoConfiguration(max_items=2))

        result = component.todo_bulk_add(
            items=[
                {"content": "Task 1", "active_form": "Task 1"},
                {"content": "Task 2", "active_form": "Task 2"},
                {"content": "Task 3", "active_form": "Task 3"},
            ]
        )
        assert result["status"] == "error"
        assert "exceed max" in result["message"]


class TestTodoReorder:
    """Tests for the todo_reorder command."""

    def test_reorder_todos(self, todo_component):
        """Reordering todos should change their order."""
        r1 = todo_component.todo_add(content="First", active_form="First")
        r2 = todo_component.todo_add(content="Second", active_form="Second")
        r3 = todo_component.todo_add(content="Third", active_form="Third")

        # Reverse the order
        result = todo_component.todo_reorder(
            ids=[r3["item"]["id"], r2["item"]["id"], r1["item"]["id"]]
        )
        assert result["status"] == "success"

        read_result = todo_component.todo_read()
        assert read_result["items"][0]["content"] == "Third"
        assert read_result["items"][1]["content"] == "Second"
        assert read_result["items"][2]["content"] == "First"

    def test_reorder_missing_ids(self, todo_component):
        """Reorder with missing IDs should fail."""
        r1 = todo_component.todo_add(content="First", active_form="First")
        todo_component.todo_add(content="Second", active_form="Second")

        result = todo_component.todo_reorder(ids=[r1["item"]["id"]])
        assert result["status"] == "error"
        assert "Missing todo IDs" in result["message"]

    def test_reorder_unknown_ids(self, todo_component):
        """Reorder with unknown IDs should fail."""
        r1 = todo_component.todo_add(content="First", active_form="First")

        result = todo_component.todo_reorder(ids=[r1["item"]["id"], "unknown_id"])
        assert result["status"] == "error"
        assert "Unknown todo IDs" in result["message"]

    def test_reorder_duplicate_ids(self, todo_component):
        """Reorder with duplicate IDs should fail."""
        r1 = todo_component.todo_add(content="First", active_form="First")
        todo_component.todo_add(content="Second", active_form="Second")

        result = todo_component.todo_reorder(ids=[r1["item"]["id"], r1["item"]["id"]])
        assert result["status"] == "error"
        assert "Duplicate" in result["message"]


class TestTodoIdIntegration:
    """Tests for ID functionality across operations."""

    def test_ids_are_unique(self, todo_component):
        """Each added todo should have a unique ID."""
        ids = set()
        for i in range(10):
            result = todo_component.todo_add(
                content=f"Task {i}", active_form=f"Task {i}"
            )
            ids.add(result["item"]["id"])

        assert len(ids) == 10

    def test_id_preserved_on_status_change(self, todo_component):
        """ID should be preserved when status changes."""
        add_result = todo_component.todo_add(content="Task", active_form="Task")
        original_id = add_result["item"]["id"]

        todo_component.todo_set_status(id=original_id, status="in_progress")
        todo_component.todo_set_status(id=original_id, status="completed")

        read_result = todo_component.todo_read()
        assert read_result["items"][0]["id"] == original_id

    def test_todo_read_includes_ids(self, todo_component):
        """todo_read should return items with IDs."""
        todo_component.todo_add(content="Task", active_form="Task")

        result = todo_component.todo_read()
        assert "id" in result["items"][0]

    def test_bulk_add_generates_ids(self, todo_component):
        """todo_bulk_add should generate IDs for items."""
        result = todo_component.todo_bulk_add(
            items=[{"content": "Task", "active_form": "Task"}]
        )
        assert "id" in result["added"][0]

        read_result = todo_component.todo_read()
        assert "id" in read_result["items"][0]


class TestIncrementalOperationsCommands:
    """Tests for incremental operations being registered as commands."""

    def test_all_incremental_commands_registered(self, todo_component):
        """All incremental commands should be registered."""
        commands = list(todo_component.get_commands())
        command_names = [c.names[0] for c in commands]

        assert "todo_add" in command_names
        assert "todo_set_status" in command_names
        assert "todo_update" in command_names
        assert "todo_delete" in command_names
        assert "todo_bulk_add" in command_names
        assert "todo_reorder" in command_names
        # todo_write is removed - incremental operations only
        assert "todo_write" not in command_names
