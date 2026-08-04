import type { ToolUIPart } from "ai";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import {
  extractToolName,
  getAnimationText,
  getToolCategory,
} from "../../tools/GenericTool/helpers";
import { type ChainCategory, getCatalogLabel } from "./toolCatalog";

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
}

function getProviderIconSrc(tool: ToolUIPart): string | undefined {
  for (const source of [tool.input, tool.output]) {
    if (source && typeof source === "object" && "provider" in source) {
      const provider = (source as { provider: unknown }).provider;
      if (typeof provider === "string" && provider.trim()) {
        return `/integrations/${provider
          .trim()
          .toLowerCase()
          .replace(/[\s-]+/g, "_")}.png`;
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
        state,
        detail,
        providerIconSrc,
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
  const phrases: string[] = [];
  for (const row of rows) {
    const phrase = CATEGORY_SUMMARY[row.category];
    if (!phrases.includes(phrase)) phrases.push(phrase);
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
