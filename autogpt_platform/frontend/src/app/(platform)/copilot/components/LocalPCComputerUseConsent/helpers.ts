interface ToolGroup {
  tools: string[];
  whole: string;
  each: Record<string, string>;
}

const TOOL_GROUPS: ToolGroup[] = [
  {
    tools: ["local_pc_screenshot"],
    whole: "Capture screenshots of your screen",
    each: {},
  },
  {
    tools: [
      "local_pc_click",
      "local_pc_type",
      "local_pc_key",
      "local_pc_scroll",
      "local_pc_cursor_position",
    ],
    whole: "Move the pointer, click, type, scroll, and send keys",
    each: {
      local_pc_click: "Move the pointer and click",
      local_pc_type: "Type text",
      local_pc_key: "Send keys and shortcuts",
      local_pc_scroll: "Scroll",
      local_pc_cursor_position: "See where the pointer is",
    },
  },
  {
    tools: ["local_pc_list_windows", "local_pc_focus_window"],
    whole: "List and focus open windows",
    each: {
      local_pc_list_windows: "List open windows",
      local_pc_focus_window: "Focus an open window",
    },
  },
  {
    tools: ["local_pc_list_apps", "local_pc_launch_app"],
    whole: "List and launch apps",
    each: {
      local_pc_list_apps: "List your apps",
      local_pc_launch_app: "Launch apps",
    },
  },
  {
    tools: ["local_pc_clipboard_read", "local_pc_clipboard_write"],
    whole: "Read and write your clipboard",
    each: {
      local_pc_clipboard_read: "Read your clipboard",
      local_pc_clipboard_write: "Write to your clipboard",
    },
  },
  {
    tools: ["local_pc_permissions_check"],
    whole: "Check operating-system permission status",
    each: {},
  },
];

const DESCRIBED_TOOLS = new Set(TOOL_GROUPS.flatMap((group) => group.tools));

export function describeComputerUseTools(tools: readonly string[]) {
  const granted = new Set(tools);
  const lines = TOOL_GROUPS.flatMap((group) => {
    const present = group.tools.filter((tool) => granted.has(tool));
    if (present.length === group.tools.length) return [group.whole];
    return present.map((tool) => group.each[tool]);
  });
  if (tools.some((tool) => !DESCRIBED_TOOLS.has(tool))) {
    lines.push("Use other computer controls your Local PC offers");
  }
  return lines;
}
