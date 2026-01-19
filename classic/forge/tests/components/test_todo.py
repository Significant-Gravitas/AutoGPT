"""Tests for TodoComponent."""

import pytest

from forge.components.todo import TodoComponent, TodoConfiguration


@pytest.fixture
def todo_component():
    """Create a fresh TodoComponent for testing."""
    return TodoComponent()


class TestTodoWrite:
    """Tests for the todo_write command."""

    def test_write_empty_list(self, todo_component):
        """Writing an empty list should succeed."""
        result = todo_component.todo_write([])
        assert result["status"] == "success"
        assert result["item_count"] == 0
        assert result["pending"] == 0
        assert result["in_progress"] == 0
        assert result["completed"] == 0

    def test_write_single_pending_todo(self, todo_component):
        """Writing a single pending todo should succeed."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Fix the bug",
                    "status": "pending",
                    "active_form": "Fixing the bug",
                }
            ]
        )
        assert result["status"] == "success"
        assert result["item_count"] == 1
        assert result["pending"] == 1
        assert result["in_progress"] == 0

    def test_write_multiple_todos(self, todo_component):
        """Writing multiple todos with different statuses should succeed."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Research patterns",
                    "status": "completed",
                    "active_form": "Researching patterns",
                },
                {
                    "content": "Implement feature",
                    "status": "in_progress",
                    "active_form": "Implementing feature",
                },
                {
                    "content": "Write tests",
                    "status": "pending",
                    "active_form": "Writing tests",
                },
            ]
        )
        assert result["status"] == "success"
        assert result["item_count"] == 3
        assert result["pending"] == 1
        assert result["in_progress"] == 1
        assert result["completed"] == 1

    def test_write_replaces_entire_list(self, todo_component):
        """Writing should replace the entire list, not append."""
        # First write
        todo_component.todo_write(
            [
                {
                    "content": "Task 1",
                    "status": "pending",
                    "active_form": "Doing task 1",
                }
            ]
        )

        # Second write should replace
        result = todo_component.todo_write(
            [
                {
                    "content": "Task 2",
                    "status": "pending",
                    "active_form": "Doing task 2",
                }
            ]
        )
        assert result["item_count"] == 1

        # Verify only Task 2 exists
        read_result = todo_component.todo_read()
        assert len(read_result["items"]) == 1
        assert read_result["items"][0]["content"] == "Task 2"

    def test_write_warns_on_multiple_in_progress(self, todo_component):
        """Writing multiple in_progress items should include a warning."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Task 1",
                    "status": "in_progress",
                    "active_form": "Doing task 1",
                },
                {
                    "content": "Task 2",
                    "status": "in_progress",
                    "active_form": "Doing task 2",
                },
            ]
        )
        assert result["status"] == "success"
        assert "warning" in result
        assert "2 tasks are in_progress" in result["warning"]

    def test_write_validates_required_content(self, todo_component):
        """Writing without content should fail."""
        result = todo_component.todo_write(
            [
                {
                    "content": "",
                    "status": "pending",
                    "active_form": "Doing something",
                }
            ]
        )
        assert result["status"] == "error"
        assert "content" in result["message"]

    def test_write_validates_required_active_form(self, todo_component):
        """Writing without active_form should fail."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Fix bug",
                    "status": "pending",
                    "active_form": "",
                }
            ]
        )
        assert result["status"] == "error"
        assert "active_form" in result["message"]

    def test_write_validates_status(self, todo_component):
        """Writing with invalid status should fail."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Fix bug",
                    "status": "invalid_status",
                    "active_form": "Fixing bug",
                }
            ]
        )
        assert result["status"] == "error"
        assert "status" in result["message"]

    def test_write_enforces_max_items(self, todo_component):
        """Writing more items than max_items should fail."""
        component = TodoComponent(config=TodoConfiguration(max_items=2))
        result = component.todo_write(
            [
                {"content": "Task 1", "status": "pending", "active_form": "Task 1"},
                {"content": "Task 2", "status": "pending", "active_form": "Task 2"},
                {"content": "Task 3", "status": "pending", "active_form": "Task 3"},
            ]
        )
        assert result["status"] == "error"
        assert "Too many items" in result["message"]


class TestTodoRead:
    """Tests for the todo_read command."""

    def test_read_empty_list(self, todo_component):
        """Reading an empty list should return empty items."""
        result = todo_component.todo_read()
        assert result["status"] == "success"
        assert result["items"] == []
        assert result["summary"]["pending"] == 0

    def test_read_after_write(self, todo_component):
        """Reading after writing should return the written items."""
        todo_component.todo_write(
            [
                {
                    "content": "Fix bug",
                    "status": "pending",
                    "active_form": "Fixing bug",
                }
            ]
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
        todo_component.todo_write(
            [
                {"content": "Task 1", "status": "pending", "active_form": "Task 1"},
                {"content": "Task 2", "status": "pending", "active_form": "Task 2"},
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
        assert len(practices) == 4
        assert any("todo_write" in p for p in practices)

    def test_get_commands(self, todo_component):
        """CommandProvider.get_commands should yield commands."""
        commands = list(todo_component.get_commands())
        command_names = [c.names[0] for c in commands]
        assert "todo_write" in command_names
        assert "todo_read" in command_names
        assert "todo_clear" in command_names

    def test_get_messages_empty_list(self, todo_component):
        """MessageProvider should not yield messages for empty list."""
        messages = list(todo_component.get_messages())
        assert len(messages) == 0

    def test_get_messages_with_todos(self, todo_component):
        """MessageProvider should include todos in LLM context."""
        todo_component.todo_write(
            [
                {
                    "content": "Implement feature",
                    "status": "in_progress",
                    "active_form": "Implementing feature",
                },
                {
                    "content": "Write tests",
                    "status": "pending",
                    "active_form": "Writing tests",
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
        component.todo_write(
            [{"content": "Task", "status": "pending", "active_form": "Task"}]
        )

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


class TestSubItems:
    """Tests for hierarchical sub-items support."""

    def test_write_with_sub_items(self, todo_component):
        """Writing todos with sub_items should succeed."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Implement feature",
                    "status": "in_progress",
                    "active_form": "Implementing feature",
                    "sub_items": [
                        {
                            "content": "Design API",
                            "status": "completed",
                            "active_form": "Designing API",
                        },
                        {
                            "content": "Write code",
                            "status": "in_progress",
                            "active_form": "Writing code",
                        },
                        {
                            "content": "Add tests",
                            "status": "pending",
                            "active_form": "Adding tests",
                        },
                    ],
                }
            ]
        )
        assert result["status"] == "success"
        assert result["item_count"] == 1

    def test_read_returns_sub_items(self, todo_component):
        """Reading should return sub_items."""
        todo_component.todo_write(
            [
                {
                    "content": "Main task",
                    "status": "in_progress",
                    "active_form": "Working on main task",
                    "sub_items": [
                        {
                            "content": "Sub task 1",
                            "status": "completed",
                            "active_form": "Doing sub task 1",
                        },
                        {
                            "content": "Sub task 2",
                            "status": "pending",
                            "active_form": "Doing sub task 2",
                        },
                    ],
                }
            ]
        )

        result = todo_component.todo_read()
        assert result["status"] == "success"
        assert len(result["items"]) == 1
        assert "sub_items" in result["items"][0]
        assert len(result["items"][0]["sub_items"]) == 2
        assert result["items"][0]["sub_items"][0]["content"] == "Sub task 1"
        assert result["items"][0]["sub_items"][0]["status"] == "completed"

    def test_nested_sub_items(self, todo_component):
        """Writing deeply nested sub_items should succeed."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Level 1",
                    "status": "in_progress",
                    "active_form": "Level 1",
                    "sub_items": [
                        {
                            "content": "Level 2",
                            "status": "pending",
                            "active_form": "Level 2",
                            "sub_items": [
                                {
                                    "content": "Level 3",
                                    "status": "pending",
                                    "active_form": "Level 3",
                                }
                            ],
                        }
                    ],
                }
            ]
        )
        assert result["status"] == "success"

        # Verify nested structure
        read_result = todo_component.todo_read()
        level1 = read_result["items"][0]
        level2 = level1["sub_items"][0]
        level3 = level2["sub_items"][0]
        assert level3["content"] == "Level 3"

    def test_sub_items_validation_error(self, todo_component):
        """Sub-items with invalid fields should fail validation."""
        result = todo_component.todo_write(
            [
                {
                    "content": "Main task",
                    "status": "pending",
                    "active_form": "Main task",
                    "sub_items": [
                        {
                            "content": "",  # Invalid: empty content
                            "status": "pending",
                            "active_form": "Sub task",
                        }
                    ],
                }
            ]
        )
        assert result["status"] == "error"
        assert "sub_items" in result["message"]

    def test_messages_include_sub_items(self, todo_component):
        """MessageProvider should format sub-items with indentation."""
        todo_component.todo_write(
            [
                {
                    "content": "Main task",
                    "status": "in_progress",
                    "active_form": "Working on main task",
                    "sub_items": [
                        {
                            "content": "Sub completed",
                            "status": "completed",
                            "active_form": "Sub completed",
                        },
                        {
                            "content": "Sub pending",
                            "status": "pending",
                            "active_form": "Sub pending",
                        },
                    ],
                }
            ]
        )

        messages = list(todo_component.get_messages())
        assert len(messages) == 1
        content = messages[0].content

        # Check parent is shown
        assert "Working on main task" in content
        # Check sub-items are shown (with their status indicators)
        assert "[x] Sub completed" in content
        assert "[ ] Sub pending" in content


class TestTodoDecompose:
    """Tests for the todo_decompose command."""

    def test_decompose_without_llm_provider(self, todo_component):
        """Decompose should fail gracefully without LLM provider."""
        todo_component.todo_write(
            [
                {
                    "content": "Complex task",
                    "status": "pending",
                    "active_form": "Complex task",
                }
            ]
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            todo_component.todo_decompose(item_index=0)
        )
        assert result["status"] == "error"
        assert "LLM provider not configured" in result["message"]

    def test_decompose_invalid_index(self, todo_component):
        """Decompose with invalid index should fail."""
        todo_component.todo_write(
            [{"content": "Task", "status": "pending", "active_form": "Task"}]
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            todo_component.todo_decompose(item_index=5)
        )
        assert result["status"] == "error"
        assert "Invalid item_index" in result["message"]

    def test_decompose_empty_list(self, todo_component):
        """Decompose on empty list should fail."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            todo_component.todo_decompose(item_index=0)
        )
        assert result["status"] == "error"

    def test_decompose_already_has_sub_items(self, todo_component):
        """Decompose should fail if item already has sub-items."""
        todo_component.todo_write(
            [
                {
                    "content": "Task with subs",
                    "status": "pending",
                    "active_form": "Task with subs",
                    "sub_items": [
                        {
                            "content": "Existing sub",
                            "status": "pending",
                            "active_form": "Existing sub",
                        }
                    ],
                }
            ]
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            todo_component.todo_decompose(item_index=0)
        )
        assert result["status"] == "error"
        assert "already has" in result["message"]

    def test_get_commands_includes_decompose(self, todo_component):
        """CommandProvider should include todo_decompose command."""
        commands = list(todo_component.get_commands())
        command_names = [c.names[0] for c in commands]
        assert "todo_decompose" in command_names
