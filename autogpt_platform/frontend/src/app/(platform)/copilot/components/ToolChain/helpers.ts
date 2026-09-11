import type { ToolUIPart } from "ai";
import { getBlockDisplayName } from "../../helpers/toolDisplay";
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
  /** The row's user-facing work (connectors, inputs, questions) renders in
   *  the card below the chain — the row is hidden but stays mounted so the
   *  card keeps its registration. */
  lifted?: boolean;
  /** A later row shows this same sub-session's card — this one keeps its
   *  row line but renders no card, so one delegation never stacks
   *  duplicate cards down the chain. */
  supersededSubSession?: boolean;
}

const SUB_SESSION_CARD_TOOLS = new Set([
  "run_sub_session",
  "delegate_to_expert",
  "handoff_to_expert",
  "get_sub_session_result",
]);

function subSessionIdOf(row: ChainRow): string | null {
  if (!row.tool || !SUB_SESSION_CARD_TOOLS.has(row.tool)) return null;
  const output = asObject(row.output);
  const sid = output?.sub_session_id;
  return typeof sid === "string" && sid ? sid : null;
}

/** Delegating opens a run; ``get_sub_session_result`` only polls one. A
 *  re-delegation reuses the same sub-session, so grouping by id alone would
 *  hide the previous run's answer. */
const SUB_SESSION_START_TOOLS = new Set([
  "run_sub_session",
  "delegate_to_expert",
  "handoff_to_expert",
]);

/** Delegate → poll → poll chains reference the same sub-session; only the
 *  LAST row per RUN keeps its card, earlier ones are marked superseded. A
 *  fresh delegation starts a new run, so the row holding the previous run's
 *  response stays readable instead of dropping out of the transcript. */
export function markSupersededSubSessionRows(rows: ChainRow[]): ChainRow[] {
  const supersededKeys = new Set<string>();
  const openRowKey = new Map<string, string>();
  for (const row of rows) {
    const sid = subSessionIdOf(row);
    if (!sid) continue;
    const open = openRowKey.get(sid);
    if (open && !SUB_SESSION_START_TOOLS.has(row.tool ?? "")) {
      supersededKeys.add(open);
    }
    openRowKey.set(sid, row.key);
  }
  if (supersededKeys.size === 0) return rows;
  return rows.map((row) =>
    supersededKeys.has(row.key) ? { ...row, supersededSubSession: true } : row,
  );
}

const ACTION_RESPONSE_TYPES = new Set([
  "setup_requirements",
  "review_required",
  "need_login",
  "trigger_config_required",
  "suggested_goal",
]);

function actionLabel(toolName: string, tool: ToolUIPart): string | null {
  const output = tool.output;
  const data = asObject(output);
  if (!data) return null;
  if (typeof data.type !== "string" || !ACTION_RESPONSE_TYPES.has(data.type)) {
    return null;
  }
  const isBlock = toolName === "run_block" || toolName === "continue_run_block";
  if (data.type === "setup_requirements") {
    const setup =
      data.setup_info && typeof data.setup_info === "object"
        ? (data.setup_info as Record<string, unknown>)
        : null;
    const name = isBlock
      ? getBlockDisplayName(tool.title, output)
      : setup?.agent_name;
    return typeof name === "string" && name.trim()
      ? `Connect ${name.trim()} to continue`
      : "Complete setup to continue";
  }
  if (data.type === "review_required") {
    const name = isBlock
      ? getBlockDisplayName(tool.title, output)
      : data.block_name;
    return typeof name === "string" && name.trim()
      ? `Review ${name.trim()}`
      : "Review this action";
  }
  if (data.type === "suggested_goal") return "Review the suggested goal";
  return typeof data.message === "string" && data.message.trim()
    ? data.message.trim()
    : "Action required";
}

/** Setup-requirements rows whose card registers with the chain and renders
 *  outside it — including run_mcp_tool, whose hidden MCPSetupCard registers
 *  an MCP row into the same connectors table. Only rows whose card registers
 *  a ChainActionEntry may be lifted: a lifted row renders off-screen, so a
 *  card that never registers would disappear entirely. */
export function isLiftedSetupRow(row: ChainRow): boolean {
  const data = asObject(row.output);
  if (!data) return false;
  return data.type === "setup_requirements" && !!asObject(data.setup_info);
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

// The compaction row owns its own progress bar and payoff frame — folding
// it into a chain would bury both behind a collapsed "summarized context".
export const COMPACTION_PART_TYPE = "tool-context_compaction";

// Hiring, raising or updating an expert is the user's call, not a step the
// model worked through — the card renders as its own message part so the
// approval never sits inside a chain that collapses on top of it.
export const EXPERT_CHANGE_TOOLS = new Set([
  "hire_expert",
  "raise_expert",
  "update_expert",
  "confirm_expert_change",
]);

export function isExpertChangePart(part: MessagePart): boolean {
  return (
    part.type.startsWith("tool-") &&
    EXPERT_CHANGE_TOOLS.has(part.type.slice("tool-".length))
  );
}

export function isChainPart(part: MessagePart): boolean {
  if (part.type === COMPACTION_PART_TYPE || isExpertChangePart(part)) {
    return false;
  }
  return part.type === "reasoning" || part.type.startsWith("tool-");
}

// Short assistant text sandwiched between tool calls is progress narration
// ("Now searching for hotels."), not an answer — fold it into the chain so
// back-to-back tool calls collapse into one group instead of splitting on
// every sentence.
const NARRATION_MAX_CHARS = 200;

function narrationText(part: MessagePart): string | null {
  if (part.type !== "text") return null;
  const text = typeof part.text === "string" ? part.text.trim() : "";
  if (!text || text.length > NARRATION_MAX_CHARS) return null;
  return text;
}

export function toChainRow(part: MessagePart, index: number): ChainRow | null {
  const narration = narrationText(part);
  if (narration !== null) {
    return {
      key: `narration-${index}`,
      category: "narration",
      text: narration,
      state: "done",
    };
  }
  if (part.type === "reasoning") {
    const isStreaming = "state" in part && part.state === "streaming";
    // A settled reasoning part with nothing in it is a "Thought it through"
    // row over an empty panel — the model never actually wrote any.
    const hasText =
      "text" in part && typeof part.text === "string" && !!part.text.trim();
    if (!isStreaming && !hasText) return null;
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
    const requiredActionLabel = actionLabel(toolName, tool);

    const data = {
      tool: toolName,
      input: stableTool.input,
      output: tool.output,
    };

    const catalogLabel = getCatalogLabel(toolName, stableTool.input, state, {
      displayName: tool.title,
      output: tool.output,
    });
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
  team: "changed the team",
  info: "checked your account",
  narration: "",
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
  const requiredAction = rows.findLast(
    (row) => row.requiresAction && !row.lifted,
  );
  if (requiredAction) return requiredAction.text;

  const phrases: string[] = [];
  const seenPhrases = new Set<string>();
  for (const row of rows) {
    if (row.category === "narration") continue;
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
  // Back-to-back expert changes (one per hire/raise call) render as one
  // group so a whole new team pages instead of stacking down the message.
  | { kind: "experts"; parts: MessagePart[]; index: number }
  | { kind: "part"; part: MessagePart; index: number };

export function buildChainSegments(
  parts: MessagePart[],
  isChainable: (part: MessagePart) => boolean = isChainPart,
): ChainSegment[] {
  const segments: ChainSegment[] = [];
  let chain: Extract<ChainSegment, { kind: "chain" }> | null = null;

  const hasChainableAhead = (from: number): boolean => {
    for (let i = from; i < parts.length; i++) {
      const part = parts[i];
      if (part.type === "step-start") continue;
      if (isChainable(part)) return true;
      if (narrationText(part) !== null) continue;
      return false;
    }
    return false;
  };

  parts.forEach((part, index) => {
    if (part.type === "step-start") return;
    if (isExpertChangePart(part)) {
      chain = null;
      const last = segments[segments.length - 1];
      if (last?.kind === "experts") last.parts.push(part);
      else segments.push({ kind: "experts", parts: [part], index });
      return;
    }
    if (isChainable(part)) {
      if (!chain) {
        chain = { kind: "chain", parts: [], index };
        segments.push(chain);
      }
      chain.parts.push(part);
      return;
    }
    // Fold short progress narration into the surrounding chain, but only
    // once a later tool call proves it was narration and not the answer.
    // Folding optimistically while streaming made every trailing answer
    // render inside the chain and then jump out to regular message text —
    // either when it outgrew NARRATION_MAX_CHARS or when the stream ended.
    if (chain && narrationText(part) !== null && hasChainableAhead(index + 1)) {
      chain.parts.push(part);
      return;
    }
    chain = null;
    segments.push({ kind: "part", part, index });
  });

  return segments;
}

/** A tool call whose result has not landed. Whatever it needs from the user
 *  has not been asked for yet. A call paused on human-in-the-loop approval
 *  is equally unresolved — only a denial or an output ends it. */
export function isToolCallPending(part: MessagePart): boolean {
  if (!part.type.startsWith("tool-")) return false;
  const state = (part as ToolUIPart).state;
  return (
    state === "input-streaming" ||
    state === "input-available" ||
    state === "approval-requested" ||
    state === "approval-responded"
  );
}
