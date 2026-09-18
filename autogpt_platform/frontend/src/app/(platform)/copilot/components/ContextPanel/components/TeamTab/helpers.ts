import type { ToolUIPart, UIMessage } from "ai";
import { asObject, str } from "../../../ToolChain/resultHelpers";

export type DelegationKind = "delegate" | "handoff" | "sub_session";

export type DelegationToolState = "running" | "done" | "error";

export interface DelegationRow {
  key: string;
  subSessionId: string | null;
  kind: DelegationKind;
  expertId: string | null;
  expertName: string | null;
  expertRole: string | null;
  expertAvatarUrl: string | null;
  /** Status frozen into the last tool output, or "running" while the tool
   *  call itself has not returned yet. */
  status: string;
  toolState: DelegationToolState;
  brief: string | null;
  response: string | null;
  lastMessage: string | null;
  errorText: string | null;
  elapsedSeconds: number | null;
  link: string | null;
  /** Times this sub-session was (re)started from the chat. */
  runs: number;
}

const START_TOOLS: Record<string, DelegationKind> = {
  delegate_to_expert: "delegate",
  handoff_to_expert: "handoff",
  run_sub_session: "sub_session",
};

const POLL_TOOL = "get_sub_session_result";

export const DELEGATION_TOOLS: ReadonlySet<string> = new Set([
  ...Object.keys(START_TOOLS),
  POLL_TOOL,
]);

const LIVE_STATUSES: ReadonlySet<string> = new Set(["running", "queued"]);

export function isLiveDelegationStatus(status: string | null): boolean {
  return LIVE_STATUSES.has(status?.toLowerCase() ?? "");
}

export function isLiveDelegation(row: DelegationRow): boolean {
  if (row.toolState === "error") return false;
  if (row.toolState === "running") return true;
  return isLiveDelegationStatus(row.status);
}

function toolNameOf(part: ToolUIPart): string {
  return part.type.replace(/^tool-/, "");
}

function toolStateOf(part: ToolUIPart): DelegationToolState {
  if (part.state === "output-error") return "error";
  if (part.state === "output-available") return "done";
  return "running";
}

function lastMessageText(output: Record<string, unknown>): string | null {
  const progress = asObject(output.progress);
  const messages = progress?.last_messages;
  if (!Array.isArray(messages)) return null;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = asObject(messages[i]);
    const content = message && str(message, "content");
    if (content) return content;
  }
  return null;
}

function applyOutput(row: DelegationRow, output: Record<string, unknown>) {
  const status = str(output, "status");
  if (status) row.status = status;
  const expert = asObject(output.expert);
  if (expert) {
    row.expertId = str(expert, "id") ?? row.expertId;
    row.expertName = str(expert, "name") ?? row.expertName;
    row.expertRole = str(expert, "role") ?? row.expertRole;
    row.expertAvatarUrl = str(expert, "avatar_url") ?? row.expertAvatarUrl;
  }
  row.response = str(output, "response") ?? row.response;
  row.lastMessage = lastMessageText(output) ?? row.lastMessage;
  row.link = str(output, "sub_autopilot_session_link") ?? row.link;
  if (typeof output.elapsed_seconds === "number") {
    row.elapsedSeconds = output.elapsed_seconds;
  }
}

function newRow(key: string, kind: DelegationKind): DelegationRow {
  return {
    key,
    subSessionId: null,
    kind,
    expertId: null,
    expertName: null,
    expertRole: null,
    expertAvatarUrl: null,
    status: "running",
    toolState: "running",
    brief: null,
    response: null,
    lastMessage: null,
    errorText: null,
    elapsedSeconds: null,
    link: null,
    runs: 0,
  };
}

function* delegationParts(
  messages: readonly UIMessage[],
): Generator<ToolUIPart> {
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const part of message.parts) {
      if (!part.type.startsWith("tool-")) continue;
      const tool = part as ToolUIPart;
      if (DELEGATION_TOOLS.has(toolNameOf(tool))) yield tool;
    }
  }
}

/** Folds every delegation tool call in the transcript into one row per
 *  sub-session, in first-seen order. Later calls (re-delegations, result
 *  polls) update their row in place, so rows never reorder while the chat
 *  streams. A poll of a sub-session whose start call is outside the loaded
 *  history still gets a row, built from the poll's own output. */
export function foldDelegations(
  messages: readonly UIMessage[],
): DelegationRow[] {
  const rows = new Map<string, DelegationRow>();
  const keyBySubSession = new Map<string, string>();

  for (const part of delegationParts(messages)) {
    const toolName = toolNameOf(part);
    const output = asObject(part.output);
    const subSessionId = output ? str(output, "sub_session_id") : null;
    const startKind = START_TOOLS[toolName];
    const existingKey = subSessionId
      ? keyBySubSession.get(subSessionId)
      : undefined;

    let row: DelegationRow;
    if (existingKey) {
      row = rows.get(existingKey)!;
    } else if (startKind || subSessionId) {
      row = newRow(part.toolCallId, startKind ?? "sub_session");
      rows.set(row.key, row);
      if (subSessionId) keyBySubSession.set(subSessionId, row.key);
    } else {
      continue;
    }

    if (subSessionId) row.subSessionId = subSessionId;
    if (startKind) {
      row.kind = startKind;
      row.runs += 1;
      const input = asObject(part.input);
      row.brief = (input && str(input, "prompt")) ?? row.brief;
      row.expertId = (input && str(input, "expert_id")) ?? row.expertId;
      row.response = null;
      row.lastMessage = null;
      row.errorText = null;
    }

    row.toolState = toolStateOf(part);
    if (row.toolState === "error") {
      row.status = "error";
      row.errorText =
        typeof part.errorText === "string" && part.errorText.trim()
          ? part.errorText.trim()
          : null;
    } else if (row.toolState === "running") {
      row.status = "running";
    } else if (output) {
      applyOutput(row, output);
    }
    if (!row.link && row.subSessionId) {
      row.link = `/copilot?sessionId=${row.subSessionId}`;
    }
  }

  return [...rows.values()];
}

export function countLiveDelegations(rows: readonly DelegationRow[]): number {
  return rows.filter(isLiveDelegation).length;
}

export function formatElapsed(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(seconds / 60);
  if (minutes === 0) return `${seconds}s`;
  const hours = Math.floor(minutes / 60);
  if (hours === 0)
    return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export type DelegationTone = "working" | "done" | "failed" | "stopped";

export function delegationTone(status: string | null): DelegationTone {
  const normalized = status?.toLowerCase() ?? "";
  if (isLiveDelegationStatus(normalized)) return "working";
  if (normalized === "completed" || normalized === "transferred") return "done";
  if (normalized === "error" || normalized === "failed") return "failed";
  return "stopped";
}

const STATUS_LABELS: Record<string, string> = {
  running: "Working",
  queued: "Working",
  completed: "Completed",
  transferred: "Handed over",
  cancelled: "Stopped",
  error: "Failed",
  failed: "Failed",
  unknown: "Status unavailable",
};

export function delegationStatusLabel(status: string | null): string {
  return STATUS_LABELS[status?.toLowerCase() ?? ""] ?? "Stopped";
}

/** What the row says under the name: live rows lead with what is happening
 *  now, settled rows with the outcome, failed rows with the error. */
export function delegationActivityText(
  row: DelegationRow,
  status: string | null,
): string | null {
  if (row.errorText) return row.errorText;
  if (isLiveDelegationStatus(status)) {
    return row.lastMessage ?? row.brief;
  }
  if (row.kind === "handoff") return row.brief;
  return row.response ?? row.lastMessage ?? row.brief;
}
