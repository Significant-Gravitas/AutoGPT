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
  | "team"
  | "info"
  | "narration";

export type ToolInput = Record<string, unknown>;

export interface ToolDisplayContext {
  displayName?: unknown;
  output?: unknown;
}

export interface ToolMeta {
  category: ChainCategory;
  running: string;
  done: string;
  subject?: (input: ToolInput, context: ToolDisplayContext) => string | null;
  /** The call's own plain-language description of the step ("Updating your
   *  launch brief"). When present it is the whole label, in place of
   *  running/done and the subject. */
  phrase?: (input: ToolInput) => string | null;
}

export function strField(input: ToolInput, key: string): string | null {
  const value = input[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function quoted(
  input: ToolInput,
  key: string,
  maxLen = 50,
): string | null {
  return quotedName(strField(input, key), maxLen);
}

const PHRASE_MAX_LEN = 80;

/** A step description the model wrote, tidied into a row label: one line,
 *  no closing full stop, capped. */
export function phraseField(input: ToolInput, key: string): string | null {
  const value = strField(input, key);
  if (!value) return null;
  const phrase = value
    .replace(/\s+/g, " ")
    .replace(/[.!]+$/, "")
    .trim();
  return phrase ? truncate(phrase, PHRASE_MAX_LEN - 1) : null;
}

export function quotedName(value: string | null, maxLen = 50): string | null {
  return value ? `"${truncate(value, maxLen)}"` : null;
}
