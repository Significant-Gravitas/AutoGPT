import type { UIDataTypes, UIMessage, UITools } from "ai";
import type { MessagePart } from "./components/ChatMessagesContainer/helpers";
import { asObject, str } from "./components/ToolChain/resultHelpers";

export type DelegationStatus =
  | "proposed"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "transferred";

/** The transcript's status corrected by the teammate's own session: they
 *  stopped on a question for the user, or the poll can no longer vouch. */
export type LiveDelegationStatus = DelegationStatus | "needs-input" | "unknown";

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
  response: string | null;
  error: string | null;
  files: DelegationFile[];
  /** The gate's review id while the hand-off waits for the user's approval. */
  reviewId: string | null;
}

const START_TOOLS = new Set(["delegate_to_expert", "handoff_to_expert"]);
const POLL_TOOL = "get_sub_session_result";

type ToolPartLike = {
  type: string;
  state?: string;
  toolCallId?: string;
  input?: unknown;
  output?: unknown;
};

function toolNameOf(part: MessagePart): string | null {
  return part.type.startsWith("tool-") ? part.type.slice(5) : null;
}

function readExpert(output: Record<string, unknown>): DelegationExpert | null {
  const expert = asObject(output.expert);
  if (!expert) return null;
  return {
    id: str(expert, "id"),
    name: str(expert, "name") ?? "Expert",
    role: str(expert, "role"),
    avatarUrl: str(expert, "avatar_url"),
    color: str(expert, "color"),
  };
}

function readFiles(output: Record<string, unknown>): DelegationFile[] {
  const raw = output.sub_workspace_files;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    const file = asObject(item);
    const name = file && str(file, "name");
    const path = file && str(file, "path");
    return name && path ? [{ name, path }] : [];
  });
}

function statusOf(
  part: ToolPartLike,
  output: Record<string, unknown> | null,
): DelegationStatus {
  if (part.state === "output-error") return "failed";
  if (!output) return "running";
  if (output.type === "approval_required") return "proposed";
  if (output.type === "error") return "failed";
  switch (str(output, "status")?.toLowerCase()) {
    case "queued":
      return "queued";
    case "completed":
      return "completed";
    case "cancelled":
      return "cancelled";
    case "error":
      return "failed";
    case "transferred":
      return "transferred";
    default:
      return "running";
  }
}

function elapsedOf(output: Record<string, unknown> | null): number | null {
  const value = output?.elapsed_seconds;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function applyOutput(
  delegation: ChatDelegation,
  part: ToolPartLike,
  output: Record<string, unknown> | null,
): ChatDelegation {
  const status = statusOf(part, output);
  return {
    ...delegation,
    expert: (output && readExpert(output)) ?? delegation.expert,
    subSessionId:
      (output && str(output, "sub_session_id")) ?? delegation.subSessionId,
    link:
      (output && str(output, "sub_autopilot_session_link")) ?? delegation.link,
    status,
    elapsedSeconds: elapsedOf(output) ?? delegation.elapsedSeconds,
    response: (output && str(output, "response")) ?? delegation.response,
    error:
      status === "failed"
        ? ((output && str(output, "error", "message")) ??
          (typeof (part as { errorText?: unknown }).errorText === "string"
            ? ((part as { errorText?: string }).errorText ?? null)
            : null) ??
          delegation.error)
        : delegation.error,
    files: output ? readFiles(output) : delegation.files,
    reviewId: status === "proposed" && output ? str(output, "review_id") : null,
  };
}

/** Walks the thread in order: each delegating call opens a delegation, and
 *  each later poll naming the same sub-session updates the most recent one.
 *  A re-delegation reuses the sub-session id, so it opens a new entry
 *  rather than overwriting the previous run's result. */
export function getChatDelegations(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): ChatDelegation[] {
  const delegations: ChatDelegation[] = [];
  const latestBySession = new Map<string, number>();
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const part of message.parts as MessagePart[]) {
      const tool = toolNameOf(part);
      if (!tool) continue;
      const toolPart = part as ToolPartLike;
      const output = asObject(toolPart.output);
      if (START_TOOLS.has(tool)) {
        const input = asObject(toolPart.input);
        const opened = applyOutput(
          {
            toolCallId: toolPart.toolCallId ?? `${tool}-${delegations.length}`,
            tool: tool as ChatDelegation["tool"],
            expertId: input ? str(input, "expert_id") : null,
            expert: null,
            prompt: input ? str(input, "prompt") : null,
            subSessionId: null,
            link: null,
            status: "running",
            elapsedSeconds: null,
            response: null,
            error: null,
            files: [],
            reviewId: null,
          },
          toolPart,
          output,
        );
        delegations.push(opened);
        if (opened.subSessionId)
          latestBySession.set(opened.subSessionId, delegations.length - 1);
        continue;
      }
      if (tool !== POLL_TOOL) continue;
      const input = asObject(toolPart.input);
      const sid =
        (output && str(output, "sub_session_id")) ??
        (input && str(input, "sub_session_id"));
      if (!sid) continue;
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
  done: number;
  failed: number;
  total: number;
}

export function countDelegations(
  delegations: ChatDelegation[],
): DelegationCounts {
  const counts: DelegationCounts = {
    proposed: 0,
    queued: 0,
    working: 0,
    done: 0,
    failed: 0,
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
      case "completed":
      case "transferred":
        counts.done += 1;
        break;
      case "failed":
      case "cancelled":
        counts.failed += 1;
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
  if (counts.queued > 0) parts.push(`${counts.queued} queued`);
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

export function delegationName(delegation: ChatDelegation): string {
  return delegation.expert?.name ?? "Expert";
}

export type DelegationTone =
  | "working"
  | "waiting"
  | "done"
  | "failed"
  | "muted";

export interface DelegationStatusView {
  label: string;
  tone: DelegationTone;
}

/** One word per state, shared by the status line, the docked bar and the
 *  Work tab so every surface calls the same thing the same name. */
export function getDelegationStatusView(
  status: LiveDelegationStatus,
): DelegationStatusView {
  switch (status) {
    case "proposed":
      return { label: "Waiting for you", tone: "waiting" };
    case "needs-input":
      return { label: "Needs you", tone: "waiting" };
    case "queued":
      return { label: "Queued", tone: "muted" };
    case "running":
      return { label: "Working", tone: "working" };
    case "completed":
      return { label: "Done", tone: "done" };
    case "transferred":
      return { label: "Handed over", tone: "done" };
    case "failed":
      return { label: "Failed", tone: "failed" };
    case "cancelled":
      return { label: "Cancelled", tone: "muted" };
    case "unknown":
      return { label: "Unclear", tone: "muted" };
  }
}
