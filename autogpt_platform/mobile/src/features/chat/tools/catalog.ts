import type { SFSymbol } from "expo-symbols";

/**
 * Presentation families from docs/engineering/native-mobile/card-catalog.md.
 * `actionable` marks families whose cards carry user actions on the web.
 * This release renders every family with the same read-only details view,
 * so an actionable card says plainly that its actions aren't in the app yet.
 */
export interface CardFamily {
  id: string;
  label: string;
  symbol: SFSymbol;
  actionable: boolean;
}

const FAMILIES = {
  status: {
    id: "status",
    label: "Status",
    symbol: "info.circle",
    actionable: false,
  },
  access: {
    id: "access",
    label: "Sign-in needed",
    symbol: "lock",
    actionable: true,
  },
  workflow: {
    id: "workflow",
    label: "Workflow",
    symbol: "bolt",
    actionable: true,
  },
  connection: {
    id: "connection",
    label: "Connection",
    symbol: "key",
    actionable: true,
  },
  construction: {
    id: "construction",
    label: "Workflow building",
    symbol: "wrench.and.screwdriver",
    actionable: true,
  },
  question: {
    id: "question",
    label: "Question",
    symbol: "questionmark.circle",
    actionable: true,
  },
  capability: {
    id: "capability",
    label: "Capability",
    symbol: "puzzlepiece.extension",
    actionable: true,
  },
  approval: {
    id: "approval",
    label: "Approval",
    symbol: "hand.raised",
    actionable: true,
  },
  schedule: {
    id: "schedule",
    label: "Schedule",
    symbol: "calendar",
    actionable: true,
  },
  expertResource: {
    id: "expertResource",
    label: "Expert resources",
    symbol: "person.2",
    actionable: true,
  },
  delegation: {
    id: "delegation",
    label: "Delegated work",
    symbol: "arrow.triangle.2.circlepath",
    actionable: true,
  },
  docs: {
    id: "docs",
    label: "Web and docs",
    symbol: "globe",
    actionable: false,
  },
  files: { id: "files", label: "Files", symbol: "folder", actionable: true },
  computer: {
    id: "computer",
    label: "Computer",
    symbol: "laptopcomputer",
    actionable: true,
  },
  memory: {
    id: "memory",
    label: "Memory and plans",
    symbol: "brain",
    actionable: false,
  },
  memoryForget: {
    id: "memoryForget",
    label: "Memory",
    symbol: "brain",
    actionable: true,
  },
  channels: {
    id: "channels",
    label: "Channels and skills",
    symbol: "text.bubble",
    actionable: false,
  },
  company: {
    id: "company",
    label: "AI company",
    symbol: "person.3",
    actionable: true,
  },
  historical: {
    id: "historical",
    label: "Earlier result",
    symbol: "clock",
    actionable: false,
  },
} satisfies Record<string, CardFamily>;

type FamilyID = keyof typeof FAMILIES;

/** Every `type` the backend's tool response models declare, by family. */
const RESPONSE_TYPES: Record<string, FamilyID> = {
  error: "status",
  no_results: "status",
  need_login: "access",
  input_validation_error: "status",
  understanding_updated: "status",
  platform_info: "status",
  agents_found: "workflow",
  agent_details: "workflow",
  execution_started: "workflow",
  agent_output: "workflow",
  setup_requirements: "connection",
  credential_grant_requested: "connection",
  trigger_config_required: "connection",
  suggested_goal: "construction",
  agent_builder_guide: "construction",
  agent_builder_preview: "construction",
  agent_builder_saved: "construction",
  agent_builder_validation_result: "construction",
  agent_builder_fix_result: "construction",
  task_decomposition: "construction",
  agent_builder_clarification_needed: "question",
  expert_onboarding: "question",
  block_list: "capability",
  block_details: "capability",
  block_output: "capability",
  capability_list: "capability",
  capability_details: "capability",
  review_required: "approval",
  approval_required: "approval",
  schedule_list: "schedule",
  schedule_deleted: "schedule",
  schedule_toggled: "schedule",
  schedule_created: "schedule",
  routines: "schedule",
  routine: "schedule",
  agent_trigger_list: "schedule",
  trigger_setup: "schedule",
  preset_list: "schedule",
  preset_updated: "schedule",
  preset_deleted: "schedule",
  expert_workflow: "expertResource",
  expert_workflows: "expertResource",
  expert_credentials: "expertResource",
  mcp_tools_discovered: "delegation",
  mcp_tool_output: "delegation",
  doc_search_results: "docs",
  doc_page: "docs",
  web_fetch: "docs",
  web_search: "docs",
  feature_request_search: "docs",
  feature_request_created: "docs",
  workspace_file_list: "files",
  workspace_file_content: "files",
  workspace_file_metadata: "files",
  workspace_file_written: "files",
  workspace_file_deleted: "files",
  folder_created: "files",
  folder_list: "files",
  folder_updated: "files",
  folder_moved: "files",
  folder_deleted: "files",
  agents_moved_to_folder: "files",
  browser_navigate: "computer",
  browser_act: "computer",
  browser_screenshot: "computer",
  bash_exec: "computer",
  desktop_stream: "computer",
  memory_store: "memory",
  memory_search: "memory",
  memory_forget_candidates: "memoryForget",
  memory_forget_confirm: "memoryForget",
  todo_write: "memory",
  chat_platform_channel_list: "channels",
  chat_platform_posted: "channels",
  chat_platform_edited: "channels",
  skill_stored: "channels",
  skill_loaded: "channels",
  skill_deleted: "channels",
  skill_list: "channels",
  expert_soul_updated: "company",
  expert_change_proposed: "company",
  expert_change_applied: "company",
  team_roster: "company",
  expert_chat_list: "company",
  expert_chat_transcript: "company",
  team_consult: "company",
  session_list: "company",
  session_message: "company",
  mcp_guide: "historical",
};

export function familyForResponseType(type: string): CardFamily | null {
  const id = RESPONSE_TYPES[type];
  return id ? FAMILIES[id] : null;
}
