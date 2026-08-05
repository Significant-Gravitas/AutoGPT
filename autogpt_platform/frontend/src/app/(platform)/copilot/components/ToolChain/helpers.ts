import type { ToolUIPart } from "ai";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import {
  extractToolName,
  getAnimationText,
  getToolCategory,
} from "../../tools/GenericTool/helpers";
import { type ChainCategory, getCatalogLabel } from "./toolCatalog";
import { asObject, integrationIconSrc } from "./resultHelpers";

export type ChainRowState = "running" | "done" | "error";

export interface ChainRow {
  key: string;
  category: ChainCategory;
  text: string;
  state: ChainRowState;
  detail?: string;
  providerIconSrc?: string;
  tool?: string;
  input?: unknown;
  output?: unknown;
  reasoningText?: string;
  requiresAction?: boolean;
}

const ACTION_RESPONSE_TYPES = new Set([
  "setup_requirements",
  "review_required",
  "need_login",
  "trigger_config_required",
  "agent_builder_clarification_needed",
  "suggested_goal",
]);

function actionLabel(output: unknown): string | null {
  const data = asObject(output);
  if (!data) return null;
  if (typeof data.type !== "string" || !ACTION_RESPONSE_TYPES.has(data.type)) {
    return null;
  }
  if (data.type === "setup_requirements") {
    const setup =
      data.setup_info && typeof data.setup_info === "object"
        ? (data.setup_info as Record<string, unknown>)
        : null;
    const name = setup?.agent_name;
    return typeof name === "string" && name.trim()
      ? `Connect ${name.trim()} to continue`
      : "Complete setup to continue";
  }
  if (data.type === "review_required") {
    const name = data.block_name;
    return typeof name === "string" && name.trim()
      ? `Review ${name.trim()}`
      : "Review this action";
  }
  if (data.type === "agent_builder_clarification_needed") {
    return "Answer a few questions";
  }
  if (data.type === "suggested_goal") return "Review the suggested goal";
  return typeof data.message === "string" && data.message.trim()
    ? data.message.trim()
    : "Action required";
}

function getProviderIconSrc(tool: ToolUIPart): string | undefined {
  for (const source of [tool.input, tool.output]) {
    if (source && typeof source === "object" && "provider" in source) {
      const provider = (source as { provider: unknown }).provider;
      if (typeof provider === "string" && provider.trim()) {
        return integrationIconSrc(provider) ?? undefined;
      }
    }
  }
  return undefined;
}

export function isChainPart(part: MessagePart): boolean {
  return part.type === "reasoning" || part.type.startsWith("tool-");
}

export function toChainRow(part: MessagePart, index: number): ChainRow | null {
  if (part.type === "reasoning") {
    const isStreaming = "state" in part && part.state === "streaming";
    return {
      key: `reasoning-${index}`,
      category: "reasoning",
      text: isStreaming ? "Thinking…" : "Thought",
      state: isStreaming ? "running" : "done",
      reasoningText:
        "text" in part && typeof part.text === "string" ? part.text : undefined,
    };
  }
  if (part.type.startsWith("tool-")) {
    const tool = part as ToolUIPart;
    const toolName = extractToolName(tool);
    const state: ChainRowState =
      tool.state === "output-error"
        ? "error"
        : tool.state === "output-available"
          ? "done"
          : "running";
    const detail =
      state === "error" && typeof tool.errorText === "string"
        ? tool.errorText
        : undefined;

    // While the input JSON is still streaming, labels/icons must not read
    // it — partial subjects re-trigger the swap animation on every delta.
    const isInputStreaming = tool.state === "input-streaming";
    const stableTool = isInputStreaming
      ? ({ ...tool, input: undefined } as ToolUIPart)
      : tool;

    const providerIconSrc = getProviderIconSrc(stableTool);
    const requiredActionLabel = actionLabel(tool.output);

    const data = {
      tool: toolName,
      input: stableTool.input,
      output: tool.output,
    };

    const catalogLabel = getCatalogLabel(toolName, stableTool.input, state);
    if (catalogLabel) {
      return {
        key: tool.toolCallId,
        ...catalogLabel,
        text: requiredActionLabel ?? catalogLabel.text,
        state,
        detail,
        providerIconSrc,
        requiresAction: requiredActionLabel !== null,
        ...data,
      };
    }

    const category = getToolCategory(toolName);
    return {
      key: tool.toolCallId,
      category,
      text: getAnimationText(stableTool, category),
      state,
      detail,
      providerIconSrc,
      requiresAction: requiredActionLabel !== null,
      ...data,
    };
  }
  return null;
}

const CATEGORY_SUMMARY: Record<ChainRow["category"], string> = {
  reasoning: "thought it through",
  bash: "ran commands",
  web: "searched the web",
  browser: "browsed the web",
  "file-read": "read files",
  "file-write": "wrote files",
  "file-delete": "deleted files",
  "file-list": "listed files",
  search: "searched files",
  edit: "edited files",
  todo: "updated tasks",
  compaction: "summarized context",
  agent: "ran agents",
  "agent-build": "built agents",
  plan: "planned the approach",
  block: "ran blocks",
  memory: "managed memory",
  folder: "organized folders",
  schedule: "managed schedules",
  trigger: "managed triggers",
  preset: "managed presets",
  chat: "posted updates",
  mcp: "used MCP tools",
  docs: "read the docs",
  skill: "used skills",
  integration: "connected integrations",
  feature: "handled feature requests",
  question: "asked you questions",
  info: "checked your account",
  other: "used tools",
};

// Streaming → live label of the running row; settled → verb-chain like
// "Updated tasks, searched the web, ran commands".
export function getChainHeading(
  rows: ChainRow[],
  isStreaming: boolean,
): string {
  if (isStreaming) {
    const running = rows.findLast((r) => r.state === "running");
    if (running) return running.text;
  }
  const error = rows.findLast((row) => row.state === "error");
  if (error) return error.detail ?? error.text;
  const requiredAction = rows.findLast((row) => row.requiresAction);
  if (requiredAction) return requiredAction.text;

  const phrases: string[] = [];
  const seenPhrases = new Set<string>();
  for (const row of rows) {
    const phrase = CATEGORY_SUMMARY[row.category];
    if (!seenPhrases.has(phrase)) {
      seenPhrases.add(phrase);
      phrases.push(phrase);
    }
    if (phrases.length === 3) break;
  }
  const joined = phrases.join(", ") || "Working…";
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}

export type ChainSegment =
  | { kind: "chain"; parts: MessagePart[]; index: number }
  | { kind: "part"; part: MessagePart; index: number };

export function buildChainSegments(
  parts: MessagePart[],
  isChainable: (part: MessagePart) => boolean = isChainPart,
): ChainSegment[] {
  const segments: ChainSegment[] = [];
  let chain: Extract<ChainSegment, { kind: "chain" }> | null = null;

  parts.forEach((part, index) => {
    if (part.type === "step-start") return;
    if (isChainable(part)) {
      if (!chain) {
        chain = { kind: "chain", parts: [], index };
        segments.push(chain);
      }
      chain.parts.push(part);
      return;
    }
    chain = null;
    segments.push({ kind: "part", part, index });
  });

  return segments;
}
