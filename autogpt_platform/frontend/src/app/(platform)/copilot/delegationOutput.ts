import type { HeldOutcome } from "./components/ChatMessagesContainer/heldCallRows";
import { asObject, str } from "./components/ToolChain/resultHelpers";
import type {
  ChatDelegation,
  DelegationExpert,
  DelegationFile,
  DelegationStatus,
} from "./delegations";

export type ToolPartLike = {
  type: string;
  state?: string;
  toolCallId?: string;
  input?: unknown;
  output?: unknown;
  errorText?: unknown;
};

const STOPPED = "Stopped";

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
    if (!name || !path) return [];
    const size = file.size_bytes ?? file.size;
    return [{ name, path, sizeBytes: typeof size === "number" ? size : null }];
  });
}

function readOptions(output: Record<string, unknown>): string[] {
  const raw = output.question_options ?? output.options;
  if (!Array.isArray(raw)) return [];
  return raw.filter((option): option is string => typeof option === "string");
}

function readNumber(output: Record<string, unknown> | null, key: string) {
  const value = output?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** A call the stream finished without a result object was cut short: the
 *  user stopped the chat (live, or `""` persisted on reload). */
function wasStopped(part: ToolPartLike, output: unknown): boolean {
  if (part.state === "output-error") return part.errorText === "Cancelled";
  return part.state === "output-available" && !asObject(output);
}

function statusOf(
  part: ToolPartLike,
  output: Record<string, unknown> | null,
): DelegationStatus {
  if (wasStopped(part, part.output)) return "cancelled";
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
    case "needs_input":
      return "needs-input";
    default:
      return "running";
  }
}

function errorOf(
  delegation: ChatDelegation,
  part: ToolPartLike,
  output: Record<string, unknown> | null,
  status: DelegationStatus,
): string | null {
  if (status === "cancelled")
    return (output && str(output, "error", "message")) ?? STOPPED;
  if (status !== "failed") return delegation.error;
  return (
    (output && str(output, "error", "message")) ??
    (typeof part.errorText === "string" ? part.errorText : null) ??
    delegation.error
  );
}

export function applyOutput(
  delegation: ChatDelegation,
  part: ToolPartLike,
  output: Record<string, unknown> | null,
): ChatDelegation {
  const status = statusOf(part, output);
  const pick = (key: string) => (output && str(output, key)) ?? null;
  return {
    ...delegation,
    expert: (output && readExpert(output)) ?? delegation.expert,
    subSessionId: pick("sub_session_id") ?? delegation.subSessionId,
    link: pick("sub_autopilot_session_link") ?? delegation.link,
    status,
    elapsedSeconds:
      readNumber(output, "elapsed_seconds") ?? delegation.elapsedSeconds,
    costUsd: readNumber(output, "cost_usd") ?? delegation.costUsd,
    startedAt: pick("started_at") ?? delegation.startedAt,
    finishedAt: pick("finished_at") ?? delegation.finishedAt,
    response: pick("response") ?? delegation.response,
    question: status === "needs-input" ? pick("question") : null,
    questionOptions:
      status === "needs-input" && output ? readOptions(output) : [],
    error: errorOf(delegation, part, output, status),
    files: output ? readFiles(output) : delegation.files,
    reviewId: status === "proposed" && output ? pick("review_id") : null,
  };
}

const TURNED_DOWN: Record<string, string> = {
  rejected: "You turned down the hand-off",
  expired: "The approval expired",
  closed: "Closed without running",
};

/** Turned down at the approval card: nothing ran, so there is no run to
 *  report on. */
export function isTurnedDown(delegation: ChatDelegation): boolean {
  return (
    delegation.status === "cancelled" &&
    !delegation.subSessionId &&
    Object.values(TURNED_DOWN).includes(delegation.error ?? "")
  );
}

/** The held call's late result, which arrives on a user row once the user
 *  answers the card: an approval carries the run's own output. */
export function applyHeldOutcome(
  delegation: ChatDelegation,
  outcome: HeldOutcome,
): ChatDelegation {
  if (outcome.outcome === "unknown") return { ...delegation, reviewId: null };
  if (outcome.outcome !== "approved") {
    return {
      ...delegation,
      status: "cancelled",
      reviewId: null,
      error: TURNED_DOWN[outcome.outcome] ?? STOPPED,
    };
  }
  const output = asObject(outcome.output);
  const approved = { ...delegation, approved: true };
  if (output)
    return applyOutput(
      approved,
      { type: `tool-${delegation.tool}`, state: "output-available", output },
      output,
    );
  const text = typeof outcome.output === "string" ? outcome.output.trim() : "";
  return text
    ? { ...approved, status: "completed", reviewId: null, response: text }
    : { ...approved, status: "cancelled", reviewId: null, error: STOPPED };
}
