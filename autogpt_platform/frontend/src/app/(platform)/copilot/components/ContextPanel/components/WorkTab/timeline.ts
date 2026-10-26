import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { toMiniRow } from "../../../ToolChain/SubSessionLive";
import type {
  ChatDelegation,
  LiveDelegationStatus,
} from "../../../../delegations";

export type TimelineKind =
  | "Approved"
  | "Action"
  | "Thought"
  | "Question"
  | "You answered"
  | "Response"
  | "Error";

export interface TimelineEntry {
  key: string;
  kind: TimelineKind;
  text: string;
  at: number | null;
  live: boolean;
}

type RawMessage = Record<string, unknown>;

function timeOf(message: RawMessage): number | null {
  const at = Date.parse(String(message.created_at ?? ""));
  return Number.isFinite(at) ? at : null;
}

function textOf(message: RawMessage): string | null {
  return typeof message.content === "string" && message.content.trim()
    ? message.content.trim()
    : null;
}

function parseArgs(raw: unknown): unknown {
  if (typeof raw !== "string") return raw ?? {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

/** Where this hand-off's run starts in the teammate's thread: the last
 *  message carrying the brief, else the thread's last user message. */
function runStart(messages: RawMessage[], brief: string | null): number {
  const probe = brief?.trim().slice(0, 40);
  const briefed = probe
    ? messages.findLastIndex(
        (m) => m.role === "user" && (textOf(m) ?? "").includes(probe),
      )
    : -1;
  if (briefed !== -1) return briefed;
  return messages.findLastIndex((m) => m.role === "user");
}

function toolEntries(message: RawMessage, index: number): TimelineEntry[] {
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  return calls.flatMap((raw, i) => {
    const call = raw as {
      display_name?: unknown;
      function?: { name?: unknown; arguments?: unknown };
    };
    const name = String(call?.function?.name ?? "").trim();
    if (!name) return [];
    const row = toMiniRow(
      {
        name,
        input: parseArgs(call.function?.arguments),
        displayName: call.display_name,
      },
      i,
      false,
    );
    return [
      {
        key: `${index}-tool-${i}`,
        kind: name === "ask_question" ? "Question" : "Action",
        text: row.text,
        at: timeOf(message),
        live: false,
      },
    ];
  });
}

/** The teammate's run as the panel tells it: each step with its time and
 *  kind, from the brief onwards. The last words are the Response once the
 *  run is over; before that they are Thoughts. */
export function buildTimeline(
  session: SessionDetailResponse | null,
  delegation: ChatDelegation,
  status: LiveDelegationStatus,
): TimelineEntry[] {
  const messages = (session?.messages ?? []) as RawMessage[];
  const start = runStart(messages, delegation.prompt);
  const entries: TimelineEntry[] = [];
  if (delegation.approved) {
    const at = Date.parse(delegation.startedAt ?? "");
    entries.push({
      key: "approved",
      kind: "Approved",
      text: "You approved the hand-off",
      at: Number.isFinite(at) ? at : null,
      live: false,
    });
  }
  messages.forEach((message, index) => {
    if (index <= start) return;
    const text = textOf(message);
    if (message.role === "user" && text)
      entries.push({
        key: `${index}-user`,
        kind: "You answered",
        text,
        at: timeOf(message),
        live: false,
      });
    if (message.role !== "assistant") return;
    if (text)
      entries.push({
        key: `${index}-text`,
        kind: "Thought",
        text,
        at: timeOf(message),
        live: false,
      });
    entries.push(...toolEntries(message, index));
  });
  const finished = status === "completed" || status === "transferred";
  const lastThought = entries.findLastIndex((e) => e.kind === "Thought");
  if (finished && lastThought !== -1)
    entries[lastThought] = { ...entries[lastThought], kind: "Response" };
  if (status === "running" && entries.length > 0)
    entries[entries.length - 1] = {
      ...entries[entries.length - 1],
      live: true,
    };
  if (status === "failed" && delegation.error)
    entries.push({
      key: "error",
      kind: "Error",
      text: delegation.error,
      at: Date.parse(delegation.finishedAt ?? "") || null,
      live: false,
    });
  return entries;
}

export function formatClock(at: number | null): string | null {
  if (at === null) return null;
  return new Date(at).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}
