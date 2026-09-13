from collections.abc import Iterable

_COARSE_TOOL_GRANTS: dict[str, frozenset[str]] = {
    "screenshot": frozenset({"local_pc_screenshot"}),
    "input": frozenset(
        {
            "local_pc_click",
            "local_pc_type",
            "local_pc_key",
            "local_pc_scroll",
            "local_pc_cursor_position",
        }
    ),
    "windows": frozenset({"local_pc_list_windows", "local_pc_focus_window"}),
    "apps": frozenset({"local_pc_list_apps", "local_pc_launch_app"}),
    "clipboard": frozenset({"local_pc_clipboard_read", "local_pc_clipboard_write"}),
    "permissions": frozenset({"local_pc_permissions_check"}),
}
_FINE_TOOL_GRANTS: dict[str, str] = {
    "screenshot.capture": "local_pc_screenshot",
    "screenshot.region": "local_pc_screenshot",
    "screenshot.window": "local_pc_screenshot",
    "input.click": "local_pc_click",
    "input.click.modifiers": "local_pc_click",
    "input.click.button": "local_pc_click",
    "input.scroll.amount": "local_pc_scroll",
    "cursor.position": "local_pc_cursor_position",
    "window.list": "local_pc_list_windows",
    "window.focus": "local_pc_focus_window",
    "app.list": "local_pc_list_apps",
    "app.launch": "local_pc_launch_app",
    "clipboard.read": "local_pc_clipboard_read",
    "clipboard.write": "local_pc_clipboard_write",
    "permissions.check": "local_pc_permissions_check",
}


def local_pc_tool_names_for_features(
    coarse_features: Iterable[str], fine_features: Iterable[str]
) -> frozenset[str]:
    coarse_tools = {
        name
        for feature in coarse_features
        for name in _COARSE_TOOL_GRANTS.get(feature, ())
    }
    fine_tools = {
        name
        for feature in fine_features
        if (name := _FINE_TOOL_GRANTS.get(feature)) is not None
    }
    return frozenset(coarse_tools | fine_tools)
