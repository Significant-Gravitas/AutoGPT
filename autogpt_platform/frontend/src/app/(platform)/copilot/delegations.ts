import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  getHeldOutcomes,
  type HeldOutcome,
} from "./components/ChatMessagesContainer/heldCallRows";
import type { MessagePart } from "./components/ChatMessagesContainer/helpers";
import { asObject, str } from "./components/ToolChain/resultHelpers";
import {
  applyHeldOutcome,
  applyOutput,
  type ToolPartLike,
} from "./delegationOutput";

export type DelegationStatus =
  | "proposed"
  | "queued"
  | "running"
  | "needs-input"
  | "completed"
  | "failed"
  | "cancelled"
  | "transferred";

/** The transcript's status corrected by the teammate's own session: they
 *  stopped on a question for the user, or the poll can no longer vouch. */
export type LiveDelegationStatus = DelegationStatus | "unknown";

export interface DelegationExpert {
  id: string | null;
  name: string;
  role: string | null;
  avatarUrl: string | null;
  color: string | null;
}

export interface DelegationFile {
  name: string;
  path: string;
  sizeBytes: number | null;
}

/** One hand-off from this chat to a teammate, as the transcript tells it:
 *  the delegating call plus every later poll of the same run folded in. */
export interface ChatDelegation {
  toolCallId: string;
  tool: "delegate_to_expert" | "handoff_to_expert";
  expertId: string | null;
  expert: DelegationExpert | null;
  prompt: string | null;
  subSessionId: string | null;
  link: string | null;
  status: DelegationStatus;
  elapsedSeconds: number | null;
  costUsd: number | null;
  startedAt: string | null;
  finishedAt: string | null;
  response: string | null;
  /** What the teammate stopped on, when the result says they need the user. */
  question: string | null;
  questionOptions: string[];
  error: string | null;
  files: DelegationFile[];
  /** The gate's review id while the hand-off waits for the user's approval. */
  reviewId: string | null;
  /** The user approved it at the gate (Ask First) before it ran. */
  approved: boolean;
}

const START_TOOLS = new Set(["delegate_to_expert", "handoff_to_expert"]);
const POLL_TOOL = "get_sub_session_result";

function toolNameOf(part: MessagePart): string | null {
  return part.type.startsWith("tool-") ? part.type.slice(5) : null;
}

function openDelegation(
  tool: string,
  toolPart: ToolPartLike,
  fallbackId: string,
): ChatDelegation {
  const input = asObject(toolPart.input);
  return applyOutput(
    {
      toolCallId: toolPart.toolCallId ?? fallbackId,
      tool: tool as ChatDelegation["tool"],
      expertId: input ? str(input, "expert_id") : null,
      expert: null,
      prompt: input ? str(input, "prompt") : null,
      subSessionId: null,
      link: null,
      status: "running",
      elapsedSeconds: null,
      costUsd: null,
      startedAt: null,
      finishedAt: null,
      response: null,
      question: null,
      questionOptions: [],
      error: null,
      files: [],
      reviewId: null,
      approved: false,
    },
    toolPart,
    asObject(toolPart.output),
  );
}

/** Walks the thread in order: each delegating call opens a delegation, and
 *  each later poll naming the same sub-session updates the most recent one.
 *  A re-delegation reuses the sub-session id, so it opens a new entry
 *  rather than overwriting the previous run's result. A hand-off held for
 *  approval takes its run's result from the user row the answer writes. */
export function getChatDelegations(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  heldOutcomes: ReadonlyMap<string, HeldOutcome> = getHeldOutcomes(messages),
): ChatDelegation[] {
  const delegations: ChatDelegation[] = [];
  const latestBySession = new Map<string, number>();
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const part of message.parts as MessagePart[]) {
      const tool = toolNameOf(part);
      if (!tool) continue;
      const toolPart = part as ToolPartLike;
      if (START_TOOLS.has(tool)) {
        let opened = openDelegation(
          tool,
          toolPart,
          `${tool}-${delegations.length}`,
        );
        const outcome = heldOutcomes.get(opened.toolCallId);
        if (opened.status === "proposed" && outcome)
          opened = applyHeldOutcome(opened, outcome);
        delegations.push(opened);
        if (opened.subSessionId)
          latestBySession.set(opened.subSessionId, delegations.length - 1);
        continue;
      }
      if (tool !== POLL_TOOL) continue;
      const output = asObject(toolPart.output);
      const input = asObject(toolPart.input);
      const sid =
        (output && str(output, "sub_session_id")) ??
        (input && str(input, "sub_session_id"));
      if (!sid || !output) continue;
      const index = latestBySession.get(sid);
      if (index === undefined) continue;
      delegations[index] = applyOutput(delegations[index], toolPart, output);
    }
  }
  return delegations;
}

export interface DelegationCounts {
  proposed: number;
  queued: number;
  working: number;
  needsInput: number;
  done: number;
  failed: number;
  cancelled: number;
  total: number;
}

export function countDelegations(
  delegations: ChatDelegation[],
): DelegationCounts {
  const counts: DelegationCounts = {
    proposed: 0,
    queued: 0,
    working: 0,
    needsInput: 0,
    done: 0,
    failed: 0,
    cancelled: 0,
    total: delegations.length,
  };
  for (const delegation of delegations) {
    switch (delegation.status) {
      case "proposed":
        counts.proposed += 1;
        break;
      case "queued":
        counts.queued += 1;
        break;
      case "running":
        counts.working += 1;
        break;
      case "needs-input":
        counts.needsInput += 1;
        break;
      case "completed":
      case "transferred":
        counts.done += 1;
        break;
      case "failed":
        counts.failed += 1;
        break;
      case "cancelled":
        counts.cancelled += 1;
        break;
    }
  }
  return counts;
}

function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** The one line the docked bar and the status line share: what the experts
 *  are doing right now, most urgent first. Null when nothing is in flight. */
export function getDelegationSummary(counts: DelegationCounts): string | null {
  const parts: string[] = [];
  if (counts.proposed > 0)
    parts.push(
      `${plural(counts.proposed, "hand-off")} waiting for your approval`,
    );
  if (counts.working > 0)
    parts.push(`${plural(counts.working, "expert")} working`);
  if (counts.queued > 0)
    parts.push(
      counts.working > 0
        ? `${counts.queued} queued`
        : `${plural(counts.queued, "expert")} queued`,
    );
  if (parts.length > 0) return parts.join(" · ");
  if (counts.failed > 0) return `${plural(counts.failed, "expert")} stopped`;
  return null;
}

export function formatElapsed(seconds: number | null): string | null {
  if (seconds === null) return null;
  const whole = Math.max(0, Math.round(seconds));
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}
