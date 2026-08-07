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
  | "info"
  | "narration";

export type ToolInput = Record<string, unknown>;

export interface ToolMeta {
  category: ChainCategory;
  running: string;
  done: string;
  subject?: (input: ToolInput) => string | null;
}

export function str(input: ToolInput, key: string): string | null {
  const value = input[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function quoted(
  input: ToolInput,
  key: string,
  maxLen = 50,
): string | null {
  const value = str(input, key);
  return value ? `"${truncate(value, maxLen)}"` : null;
}
