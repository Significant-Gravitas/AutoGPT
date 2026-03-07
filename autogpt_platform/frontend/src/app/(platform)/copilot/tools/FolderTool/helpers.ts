import type { ToolUIPart } from "ai";

interface FolderAgentSummary {
  id: string;
  name: string;
  description?: string;
}

interface FolderInfo {
  id: string;
  name: string;
  parent_id?: string | null;
  icon?: string | null;
  color?: string | null;
  agent_count: number;
  subfolder_count: number;
  agents?: FolderAgentSummary[] | null;
}

interface FolderTreeInfo extends FolderInfo {
  children: FolderTreeInfo[];
}

interface FolderCreatedOutput {
  type: "folder_created";
  message: string;
  folder: FolderInfo;
}

interface FolderListOutput {
  type: "folder_list";
  message: string;
  folders?: FolderInfo[];
  tree?: FolderTreeInfo[];
  count: number;
}

interface FolderUpdatedOutput {
  type: "folder_updated";
  message: string;
  folder: FolderInfo;
}

interface FolderMovedOutput {
  type: "folder_moved";
  message: string;
  folder: FolderInfo;
  target_parent_id?: string | null;
}

interface FolderDeletedOutput {
  type: "folder_deleted";
  message: string;
  folder_id: string;
}

interface AgentsMovedOutput {
  type: "agents_moved_to_folder";
  message: string;
  agent_ids: string[];
  folder_id?: string | null;
  count: number;
}

interface ErrorOutput {
  type: "error";
  message: string;
  error?: string;
}

export type FolderToolOutput =
  | FolderCreatedOutput
  | FolderListOutput
  | FolderUpdatedOutput
  | FolderMovedOutput
  | FolderDeletedOutput
  | AgentsMovedOutput
  | ErrorOutput;

export type { FolderAgentSummary, FolderInfo, FolderTreeInfo };

function parseOutput(output: unknown): FolderToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    const obj = output as Record<string, unknown>;
    if (typeof obj.type === "string") {
      return output as FolderToolOutput;
    }
  }
  return null;
}

export function getFolderToolOutput(part: {
  output?: unknown;
}): FolderToolOutput | null {
  return parseOutput(part.output);
}

export function isFolderCreated(o: FolderToolOutput): o is FolderCreatedOutput {
  return o.type === "folder_created";
}

export function isFolderList(o: FolderToolOutput): o is FolderListOutput {
  return o.type === "folder_list";
}

export function isFolderUpdated(o: FolderToolOutput): o is FolderUpdatedOutput {
  return o.type === "folder_updated";
}

export function isFolderMoved(o: FolderToolOutput): o is FolderMovedOutput {
  return o.type === "folder_moved";
}

export function isFolderDeleted(o: FolderToolOutput): o is FolderDeletedOutput {
  return o.type === "folder_deleted";
}

export function isAgentsMoved(o: FolderToolOutput): o is AgentsMovedOutput {
  return o.type === "agents_moved_to_folder";
}

export function isErrorOutput(o: FolderToolOutput): o is ErrorOutput {
  return o.type === "error";
}

export function getAnimationText(part: {
  type: string;
  state: ToolUIPart["state"];
  output?: unknown;
}): string {
  const toolName = part.type.replace(/^tool-/, "");

  switch (part.state) {
    case "input-streaming":
    case "input-available": {
      switch (toolName) {
        case "create_folder":
          return "Creating folder…";
        case "list_folders":
          return "Loading folders…";
        case "update_folder":
          return "Updating folder…";
        case "move_folder":
          return "Moving folder…";
        case "delete_folder":
          return "Deleting folder…";
        case "move_agents_to_folder":
          return "Moving agents…";
        default:
          return "Managing folders…";
      }
    }
    case "output-available": {
      const output = getFolderToolOutput(part);
      if (!output) return "Done";
      if (isErrorOutput(output)) return "Folder operation failed";
      return output.message;
    }
    case "output-error":
      return "Folder operation failed";
    default:
      return "Managing folders…";
  }
}
