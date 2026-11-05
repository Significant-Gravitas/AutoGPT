interface ToolGroup {
  tools: string[];
  line: string;
}

// cua-driver's tools, as the Local PC offers them, in the words a person
// approving computer use reads. A group is described when any of its tools
// is offered.
const TOOL_GROUPS: ToolGroup[] = [
  {
    tools: [
      "list_apps",
      "list_windows",
      "get_window_state",
      "get_desktop_state",
      "zoom",
      "verify_state",
      "get_screen_size",
      "get_cursor_position",
    ],
    line: "See your open apps and windows, read what they show, and take screenshots of them",
  },
  {
    tools: [
      "click",
      "double_click",
      "right_click",
      "type_text",
      "press_key",
      "hotkey",
      "scroll",
      "drag",
      "set_value",
      "move_cursor",
      "mouse_button_down",
      "mouse_button_up",
      "mouse_drag",
      "invoke_menu",
      "run_steps",
    ],
    line: "Click, type, scroll, and use menus in your apps",
  },
  {
    tools: ["run_script"],
    line: "Run scripts that drive your apps, once you approve each one",
  },
  {
    tools: ["set_window_frame", "bring_to_front"],
    line: "Move, resize, and bring forward your windows",
  },
  { tools: ["launch_app"], line: "Open apps and links" },
  {
    tools: [
      "get_browser_state",
      "browser_prepare",
      "browser_navigate",
      "browser_click",
      "browser_type",
      "browser_select",
      "browser_pointer",
      "browser_dialog",
    ],
    line: "Control web pages in a browser it opens with its own profile",
  },
  {
    tools: ["browser_set_input_files"],
    line: "Give web pages files from your computer to upload, once you approve each one",
  },
  {
    tools: ["clipboard_read"],
    line: "Read your clipboard, within the limits set on that computer",
  },
  {
    tools: ["clipboard_write"],
    line: "Put text on your clipboard, or a file from your computer once you approve it",
  },
];

// Tools that only report on the driver itself.
const UNDESCRIBED_TOOLS = new Set(["check_permissions", "get_guide"]);

const DESCRIBED_TOOLS = new Set(TOOL_GROUPS.flatMap((group) => group.tools));

export function describeComputerUseTools(tools: readonly string[]) {
  const offered = new Set(tools);
  const lines = TOOL_GROUPS.filter((group) =>
    group.tools.some((tool) => offered.has(tool)),
  ).map((group) => group.line);
  if (
    tools.some(
      (tool) => !DESCRIBED_TOOLS.has(tool) && !UNDESCRIBED_TOOLS.has(tool),
    )
  ) {
    lines.push("Use other computer controls your Local PC offers");
  }
  return lines;
}
