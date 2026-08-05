import { quoted, type ToolMeta } from "./toolCatalog.shared";

export const AGENT_TOOL_CATALOG: Record<string, ToolMeta> = {
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
};
