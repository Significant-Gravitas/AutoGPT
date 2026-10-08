import type { AgentStatus } from "@/components/molecules/AgentStatusAvatar/helpers";
import type { ChatStatus, ToolUIPart, UIMessage } from "ai";
import type { MessagePart } from "../components/ChatMessagesContainer/helpers";
import { isHeldCallRow } from "../components/ChatMessagesContainer/heldCallRows";
import {
  buildChainSegments,
  COMPACTION_PART_TYPE,
  getChainHeading,
  isToolCallPending,
  toChainRow,
  type ChainRow,
} from "../components/ToolChain/helpers";
import { hideKickoffMessages } from "../expertKickoff";
import { isBookkeepingPart } from "../messageParts";
import {
  getAskQuestionOutput,
  isClarificationOutput,
} from "../tools/AskQuestion/helpers";

const CARD_TOOLS = new Set(["tool-ask_question", "tool-expert_onboarding"]);
const HIDDEN_PARTS = new Set([
  "step-start",
  "reasoning",
  COMPACTION_PART_TYPE,
  "tool-TodoWrite",
]);
const USER_ACTION_TYPES = new Set([
  "setup_requirements",
  "review_required",
  "need_login",
  "trigger_config_required",
  "suggested_goal",
]);

export type CompactBlock =
  | { kind: "text"; part: MessagePart; index: number }
  | { kind: "card"; part: MessagePart; index: number }
  | { kind: "activity"; parts: MessagePart[]; index: number }
  | { kind: "experts"; parts: MessagePart[]; index: number };

export interface ActivitySummary {
  heading: string;
  rows: ChainRow[];
  state: "running" | "done" | "error";
}

function isToolPart(part: MessagePart) {
  return part.type.startsWith("tool-");
}

function outputType(part: MessagePart): string | null {
  if (!("output" in part) || !part.output) return null;
  let output: unknown = part.output;
  if (typeof output === "string") {
    try {
      output = JSON.parse(output);
    } catch {
      return null;
    }
  }
  if (!output || typeof output !== "object") return null;
  const type = (output as { type?: unknown }).type;
  return typeof type === "string" ? type : null;
}

function hasTextContent(part: MessagePart) {
  return part.type === "text" && "text" in part && !!part.text.trim();
}

function needsUserAction(part: MessagePart) {
  if (!isToolPart(part)) return false;
  if (part.type === "tool-ask_question") {
    const output = getAskQuestionOutput(part as ToolUIPart);
    return !!output && isClarificationOutput(output);
  }
  return USER_ACTION_TYPES.has(outputType(part) ?? "");
}

function isActivityPart(part: MessagePart) {
  return (
    isToolPart(part) && !CARD_TOOLS.has(part.type) && !needsUserAction(part)
  );
}

function isVisiblePart(part: MessagePart) {
  if (HIDDEN_PARTS.has(part.type) || isBookkeepingPart(part)) return false;
  return hasTextContent(part) || isToolPart(part);
}

export function buildCompactBlocks(parts: MessagePart[]): CompactBlock[] {
  return buildChainSegments(parts.filter(isVisiblePart), isActivityPart).map(
    (segment): CompactBlock => {
      if (segment.kind === "chain") {
        return { kind: "activity", parts: segment.parts, index: segment.index };
      }
      if (segment.kind === "experts") return segment;
      return {
        kind: segment.part.type === "text" ? "text" : "card",
        part: segment.part,
        index: segment.index,
      };
    },
  );
}

export function summarizeActivity(
  parts: MessagePart[],
  isStreaming: boolean,
): ActivitySummary {
  const rows = parts
    .map((part, index) => toChainRow(part, index))
    .filter((row): row is ChainRow => row !== null);
  const isRunning = isStreaming && rows.some((row) => row.state === "running");
  const hasError = rows.some((row) => row.state === "error");
  const toolRows = rows.filter((row) => row.category !== "narration");
  const heading =
    toolRows.length === 1
      ? toolRows[0].text
      : getChainHeading(rows, isStreaming);
  return {
    heading,
    rows,
    state: isRunning ? "running" : hasError ? "error" : "done",
  };
}

export function getVisibleMessages<T extends UIMessage>(messages: T[]): T[] {
  return hideKickoffMessages(messages).filter(
    (message) =>
      !isHeldCallRow(message) &&
      (message.role === "user" || message.role === "assistant"),
  );
}

export function getUserText(message: UIMessage) {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();
}

function lastContentPart(message: UIMessage) {
  return message.parts.findLast(
    (part) => part.type !== "step-start" && !isBookkeepingPart(part),
  ) as MessagePart | undefined;
}

interface DeriveArgs {
  status: ChatStatus;
  messages: UIMessage[];
  hasPendingReviews: boolean;
  isReconnecting?: boolean;
}

export function deriveAgentStatus({
  status,
  messages,
  hasPendingReviews,
  isReconnecting = false,
}: DeriveArgs): AgentStatus {
  if (hasPendingReviews) return "waiting";
  const last = messages.at(-1);
  const isInFlight =
    status === "submitted" || status === "streaming" || isReconnecting;

  if (isInFlight) {
    if (!last || last.role !== "assistant") return "thinking";
    const tail = lastContentPart(last);
    if (!tail || tail.type === "reasoning") return "thinking";
    if (isToolPart(tail) && isToolCallPending(tail)) return "working";
    return hasTextContent(tail) ? "working" : "thinking";
  }

  if (!last) return "idle";
  if (last.role !== "assistant") return "idle";
  if ((last.parts as MessagePart[]).some(needsUserAction)) return "waiting";
  return "done";
}

export function describeAgentActivity(
  agentStatus: AgentStatus,
  messages: UIMessage[],
): string | null {
  if (agentStatus !== "working") return null;
  const last = messages.at(-1);
  if (!last || last.role !== "assistant") return null;
  const tail = lastContentPart(last);
  if (!tail || !isToolPart(tail)) return null;
  return toChainRow(tail, 0)?.text ?? null;
}
