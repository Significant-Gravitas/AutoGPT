import {
  quoted,
  str,
  type ToolInput,
  type ToolMeta,
} from "./toolCatalog.shared";

export const PLATFORM_TOOL_CATALOG: Record<string, ToolMeta> = {
  TodoWrite: {
    category: "todo",
    running: "Updating tasks",
    done: "Updated tasks",
  },
  web_search: {
    category: "web",
    running: "Searching the web for",
    done: "Searched the web for",
    subject: (input) => quoted(input, "query"),
  },
  web_fetch: {
    category: "web",
    running: "Fetching",
    done: "Fetched",
    subject: (input) => quoted(input, "url", 60),
  },
  browser_navigate: {
    category: "browser",
    running: "Opening",
    done: "Opened",
    subject: (input) => quoted(input, "url", 60),
  },
  browser_act: {
    category: "browser",
    running: "Using the browser",
    done: "Used the browser",
  },
  browser_screenshot: {
    category: "browser",
    running: "Taking a screenshot",
    done: "Took a screenshot",
  },
  bash_exec: {
    category: "bash",
    running: "Running command",
    done: "Ran command",
    subject: (input) => quoted(input, "command", 60),
  },
  list_workspace_files: {
    category: "file-list",
    running: "Listing workspace files",
    done: "Listed workspace files",
  },
  read_workspace_file: {
    category: "file-read",
    running: "Reading",
    done: "Read",
    subject: (input) =>
      quoted(input, "path", 60) ?? quoted(input, "filename", 60),
  },
  write_workspace_file: {
    category: "file-write",
    running: "Writing",
    done: "Wrote",
    subject: (input) =>
      quoted(input, "path", 60) ?? quoted(input, "filename", 60),
  },
  delete_workspace_file: {
    category: "file-delete",
    running: "Deleting",
    done: "Deleted",
    subject: (input) =>
      quoted(input, "path", 60) ?? quoted(input, "filename", 60),
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
  schedule_agent: {
    category: "agent",
    running: "Scheduling agent",
    done: "Scheduled agent",
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
