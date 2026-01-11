"""
Standalone tests for pin name sanitization that can run without full backend dependencies.

These tests verify the core sanitization logic independently of the full system.
Run with: python -m pytest test_pin_sanitization_standalone.py -v
Or simply: python test_pin_sanitization_standalone.py
"""

import re
from typing import Any


# Simulate the exact cleanup function from SmartDecisionMakerBlock
def cleanup(s: str) -> str:
    """Clean up names for use as tool function names."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s).lower()


# Simulate the key parts of parse_execution_output
def simulate_tool_routing(
    emit_key: str,
    sink_node_id: str,
    sink_pin_name: str,
) -> bool:
    """
    Simulate the routing comparison from parse_execution_output.

    Returns True if routing would succeed, False otherwise.
    """
    if not emit_key.startswith("tools_^_") or "_~_" not in emit_key:
        return False

    # Extract routing info from emit key: tools_^_{node_id}_~_{field}
    selector = emit_key[8:]  # Remove "tools_^_"
    target_node_id, target_input_pin = selector.split("_~_", 1)

    # Current (buggy) comparison - direct string comparison
    return target_node_id == sink_node_id and target_input_pin == sink_pin_name


def simulate_fixed_tool_routing(
    emit_key: str,
    sink_node_id: str,
    sink_pin_name: str,
) -> bool:
    """
    Simulate the FIXED routing comparison.

    The fix: sanitize sink_pin_name before comparison.
    """
    if not emit_key.startswith("tools_^_") or "_~_" not in emit_key:
        return False

    selector = emit_key[8:]
    target_node_id, target_input_pin = selector.split("_~_", 1)

    # Fixed comparison - sanitize sink_pin_name
    return target_node_id == sink_node_id and target_input_pin == cleanup(sink_pin_name)


class TestCleanupFunction:
    """Tests for the cleanup function."""

    def test_spaces_to_underscores(self):
        assert cleanup("Max Keyword Difficulty") == "max_keyword_difficulty"

    def test_mixed_case_to_lowercase(self):
        assert cleanup("MaxKeywordDifficulty") == "maxkeyworddifficulty"

    def test_special_chars_to_underscores(self):
        assert cleanup("field@name!") == "field_name_"
        assert cleanup("CPC ($)") == "cpc____"

    def test_preserves_valid_chars(self):
        assert cleanup("valid_name-123") == "valid_name-123"

    def test_empty_string(self):
        assert cleanup("") == ""

    def test_consecutive_spaces(self):
        assert cleanup("a   b") == "a___b"

    def test_unicode(self):
        assert cleanup("café") == "caf_"


class TestCurrentRoutingBehavior:
    """Tests demonstrating the current (buggy) routing behavior."""

    def test_exact_match_works(self):
        """When names match exactly, routing works."""
        emit_key = "tools_^_node-123_~_query"
        assert simulate_tool_routing(emit_key, "node-123", "query") is True

    def test_spaces_cause_failure(self):
        """When sink_pin has spaces, routing fails."""
        sanitized = cleanup("Max Keyword Difficulty")
        emit_key = f"tools_^_node-123_~_{sanitized}"
        assert simulate_tool_routing(emit_key, "node-123", "Max Keyword Difficulty") is False

    def test_special_chars_cause_failure(self):
        """When sink_pin has special chars, routing fails."""
        sanitized = cleanup("CPC ($)")
        emit_key = f"tools_^_node-123_~_{sanitized}"
        assert simulate_tool_routing(emit_key, "node-123", "CPC ($)") is False


class TestFixedRoutingBehavior:
    """Tests demonstrating the fixed routing behavior."""

    def test_exact_match_still_works(self):
        """When names match exactly, routing still works."""
        emit_key = "tools_^_node-123_~_query"
        assert simulate_fixed_tool_routing(emit_key, "node-123", "query") is True

    def test_spaces_work_with_fix(self):
        """With the fix, spaces in sink_pin work."""
        sanitized = cleanup("Max Keyword Difficulty")
        emit_key = f"tools_^_node-123_~_{sanitized}"
        assert simulate_fixed_tool_routing(emit_key, "node-123", "Max Keyword Difficulty") is True

    def test_special_chars_work_with_fix(self):
        """With the fix, special chars in sink_pin work."""
        sanitized = cleanup("CPC ($)")
        emit_key = f"tools_^_node-123_~_{sanitized}"
        assert simulate_fixed_tool_routing(emit_key, "node-123", "CPC ($)") is True


class TestBugReproduction:
    """Exact reproduction of the reported bug."""

    def test_max_keyword_difficulty_bug(self):
        """
        Reproduce the exact bug from the issue:

        "For this agent specifically the input pin has space and unsanitized,
        the frontend somehow connect without sanitizing creating a link like:
        tools_^_767682f5-..._~_Max Keyword Difficulty
        but what's produced by backend is
        tools_^_767682f5-..._~_max_keyword_difficulty
        so the tool calls go into the void"
        """
        node_id = "767682f5-fake-uuid"
        original_field = "Max Keyword Difficulty"
        sanitized_field = cleanup(original_field)

        # What backend produces (emit key)
        emit_key = f"tools_^_{node_id}_~_{sanitized_field}"
        assert emit_key == f"tools_^_{node_id}_~_max_keyword_difficulty"

        # What frontend link has (sink_pin_name)
        frontend_sink = original_field

        # Current behavior: FAILS
        assert simulate_tool_routing(emit_key, node_id, frontend_sink) is False

        # With fix: WORKS
        assert simulate_fixed_tool_routing(emit_key, node_id, frontend_sink) is True


class TestCommonFieldNamePatterns:
    """Test common field name patterns that could cause issues."""

    FIELD_NAMES = [
        "Max Keyword Difficulty",
        "Search Volume (Monthly)",
        "CPC ($)",
        "User's Input",
        "Target URL",
        "API Response",
        "Query #1",
        "First Name",
        "Last Name",
        "Email Address",
        "Phone Number",
        "Total Cost ($)",
        "Discount (%)",
        "Created At",
        "Updated At",
        "Is Active",
    ]

    def test_current_behavior_fails_for_special_names(self):
        """Current behavior fails for names with spaces/special chars."""
        failed = []
        for name in self.FIELD_NAMES:
            sanitized = cleanup(name)
            emit_key = f"tools_^_node_~_{sanitized}"
            if not simulate_tool_routing(emit_key, "node", name):
                failed.append(name)

        # All names with spaces should fail
        names_with_spaces = [n for n in self.FIELD_NAMES if " " in n or any(c in n for c in "()$%#'")]
        assert set(failed) == set(names_with_spaces)

    def test_fixed_behavior_works_for_all_names(self):
        """Fixed behavior works for all names."""
        for name in self.FIELD_NAMES:
            sanitized = cleanup(name)
            emit_key = f"tools_^_node_~_{sanitized}"
            assert simulate_fixed_tool_routing(emit_key, "node", name) is True, f"Failed for: {name}"


def run_tests():
    """Run all tests manually without pytest."""
    import traceback

    test_classes = [
        TestCleanupFunction,
        TestCurrentRoutingBehavior,
        TestFixedRoutingBehavior,
        TestBugReproduction,
        TestCommonFieldNamePatterns,
    ]

    total = 0
    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        for name in dir(instance):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, name)()
                    print(f"  ✓ {name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
