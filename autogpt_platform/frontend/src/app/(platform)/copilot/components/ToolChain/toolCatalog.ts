import { type ToolCategory, truncate } from "../../tools/GenericTool/helpers";

export type ChainCategory =
  | ToolCategory
  | "reasoning"
  | "agent-build"
  | "plan"
  | "block"
  | "memory"
  | "folder"
  | "schedule"
  | "trigger"
  | "preset"
  | "chat"
  | "mcp"
  | "docs"
  | "skill"
  | "integration"
  | "feature"
  | "question"
  | "info";

type ToolInput = Record<string, unknown>;

interface ToolMeta {
  category: ChainCategory;
  running: string;
  done: string;
  subject?: (input: ToolInput) => string | null;
}

function str(input: ToolInput, key: string): string | null {
  const value = input[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function quoted(input: ToolInput, key: string, maxLen = 50): string | null {
  const value = str(input, key);
  return value ? `"${truncate(value, maxLen)}"` : null;
}

// Tools absent here (web_search, bash_exec, workspace files, browser,
// TodoWrite, SDK built-ins) fall back to GenericTool's labels.
export const COPILOT_TOOL_CATALOG: Record<string, ToolMeta> = {
  add_understanding: {
    category: "memory",
    running: "Noting context",
    done: "Noted context",
  },
  create_agent: {
    category: "agent-build",
    running: "Creating agent",
    done: "Created agent",
  },
  customize_agent: {
    category: "agent-build",
    running: "Customizing agent",
    done: "Customized agent",
  },
  edit_agent: {
    category: "agent-build",
    running: "Editing agent",
    done: "Edited agent",
  },
  validate_agent_graph: {
    category: "agent-build",
    running: "Validating agent",
    done: "Validated agent",
  },
  fix_agent_graph: {
    category: "agent-build",
    running: "Fixing agent",
    done: "Fixed agent",
  },
  enter_agent_building_mode: {
    category: "agent-build",
    running: "Entering building mode",
    done: "Entered building mode",
  },
  get_agent_building_guide: {
    category: "docs",
    running: "Reading the agent building guide",
    done: "Read the agent building guide",
  },
  decompose_goal: {
    category: "plan",
    running: "Breaking down the goal",
    done: "Broke down the goal",
    subject: (input) => quoted(input, "goal"),
  },
  find_block: {
    category: "block",
    running: "Finding blocks for",
    done: "Found blocks for",
    subject: (input) => quoted(input, "query"),
  },
  find_agent: {
    category: "agent",
    running: "Finding agents for",
    done: "Found agents for",
    subject: (input) => quoted(input, "query"),
  },
  find_library_agent: {
    category: "agent",
    running: "Searching your library for",
    done: "Searched your library for",
    subject: (input) => quoted(input, "query"),
  },
  memory_search: {
    category: "memory",
    running: "Searching memory for",
    done: "Searched memory for",
    subject: (input) => quoted(input, "query"),
  },
  memory_store: {
    category: "memory",
    running: "Storing memory",
    done: "Stored memory",
    subject: (input) => quoted(input, "name"),
  },
  memory_forget_search: {
    category: "memory",
    running: "Finding memories to forget",
    done: "Found memories to forget",
  },
  memory_forget_confirm: {
    category: "memory",
    running: "Forgetting memories",
    done: "Forgot memories",
  },
  create_folder: {
    category: "folder",
    running: "Creating folder",
    done: "Created folder",
    subject: (input) => quoted(input, "name"),
  },
  list_folders: {
    category: "folder",
    running: "Listing folders",
    done: "Listed folders",
  },
  update_folder: {
    category: "folder",
    running: "Updating folder",
    done: "Updated folder",
    subject: (input) => quoted(input, "name"),
  },
  move_folder: {
    category: "folder",
    running: "Moving folder",
    done: "Moved folder",
  },
  delete_folder: {
    category: "folder",
    running: "Deleting folder",
    done: "Deleted folder",
  },
  move_agents_to_folder: {
    category: "folder",
    running: "Moving agents to folder",
    done: "Moved agents to folder",
  },
  run_agent: {
    category: "agent",
    running: "Running agent",
    done: "Ran agent",
    subject: (input) =>
      quoted(input, "username_agent_slug") ??
      quoted(input, "library_agent_id", 20),
  },
  view_agent_output: {
    category: "agent",
    running: "Viewing agent output",
    done: "Viewed agent output",
  },
  run_sub_session: {
    category: "agent",
    running: "Delegating to sub-AutoPilot:",
    done: "Sub-AutoPilot handled:",
    subject: (input) => quoted(input, "prompt", 45),
  },
  get_sub_session_result: {
    category: "agent",
    running: "Checking sub-AutoPilot result",
    done: "Sub-AutoPilot result received",
  },
  list_schedules: {
    category: "schedule",
    running: "Listing schedules",
    done: "Listed schedules",
  },
  delete_schedule: {
    category: "schedule",
    running: "Deleting schedule",
    done: "Deleted schedule",
  },
  schedule_followup: {
    category: "schedule",
    running: "Scheduling a follow-up",
    done: "Scheduled a follow-up",
  },
  post_to_chat_platform: {
    category: "chat",
    running: "Posting to",
    done: "Posted to",
    subject: (input) => quoted(input, "channel"),
  },
  list_chat_platform_channels: {
    category: "chat",
    running: "Listing chat channels",
    done: "Listed chat channels",
  },
  list_agent_triggers: {
    category: "trigger",
    running: "Listing agent triggers",
    done: "Listed agent triggers",
  },
  setup_agent_webhook_trigger: {
    category: "trigger",
    running: "Setting up webhook trigger",
    done: "Set up webhook trigger",
  },
  list_presets: {
    category: "preset",
    running: "Listing presets",
    done: "Listed presets",
  },
  update_preset: {
    category: "preset",
    running: "Updating preset",
    done: "Updated preset",
  },
  delete_preset: {
    category: "preset",
    running: "Deleting preset",
    done: "Deleted preset",
  },
  run_block: {
    category: "block",
    running: "Running block",
    done: "Ran block",
    subject: (input) =>
      quoted(input, "block_name") ?? quoted(input, "block_id", 20),
  },
  continue_run_block: {
    category: "block",
    running: "Continuing block run",
    done: "Continued block run",
  },
  run_mcp_tool: {
    category: "mcp",
    running: "Running MCP tool",
    done: "Ran MCP tool",
    subject: (input) => quoted(input, "tool_name"),
  },
  get_mcp_guide: {
    category: "docs",
    running: "Reading the MCP guide",
    done: "Read the MCP guide",
  },
  search_docs: {
    category: "docs",
    running: "Searching docs for",
    done: "Searched docs for",
    subject: (input) => quoted(input, "query"),
  },
  get_doc_page: {
    category: "docs",
    running: "Reading doc page",
    done: "Read doc page",
    subject: (input) => str(input, "path"),
  },
  store_skill: {
    category: "skill",
    running: "Saving skill",
    done: "Saved skill",
    subject: (input) => quoted(input, "name"),
  },
  read_skill: {
    category: "skill",
    running: "Reading skill",
    done: "Read skill",
    subject: (input) => quoted(input, "name"),
  },
  delete_skill: {
    category: "skill",
    running: "Deleting skill",
    done: "Deleted skill",
    subject: (input) => quoted(input, "name"),
  },
  list_skills: {
    category: "skill",
    running: "Listing skills",
    done: "Listed skills",
  },
  connect_integration: {
    category: "integration",
    running: "Connecting",
    done: "Connected",
    subject: (input) => str(input, "provider"),
  },
  search_feature_requests: {
    category: "feature",
    running: "Searching feature requests for",
    done: "Searched feature requests for",
    subject: (input) => quoted(input, "query"),
  },
  create_feature_request: {
    category: "feature",
    running: "Filing feature request",
    done: "Filed feature request",
    subject: (input) => quoted(input, "title"),
  },
  get_platform_info: {
    category: "info",
    running: "Checking platform info",
    done: "Checked platform info",
  },
  ask_question: {
    category: "question",
    running: "Asking you a question",
    done: "Asked you a question",
    subject: (input) => {
      const direct = quoted(input, "question", 60);
      if (direct) return direct;
      const first = Array.isArray(input.questions)
        ? (input.questions as unknown[])[0]
        : null;
      return first && typeof first === "object"
        ? quoted(first as ToolInput, "question", 60)
        : null;
    },
  },
};

export function getCatalogLabel(
  toolName: string,
  input: unknown,
  state: "running" | "done" | "error",
): { category: ChainCategory; text: string } | null {
  const meta = COPILOT_TOOL_CATALOG[toolName];
  if (!meta) return null;
  const subject =
    meta.subject?.(
      input && typeof input === "object" ? (input as ToolInput) : {},
    ) ?? null;
  const suffix = subject ? ` ${subject}` : "";
  const text =
    state === "running"
      ? `${meta.running}${suffix}…`
      : state === "error"
        ? `Couldn't ${meta.running.charAt(0).toLowerCase()}${meta.running.slice(1)}${suffix}`
        : `${meta.done}${suffix}`;
  return { category: meta.category, text };
}
