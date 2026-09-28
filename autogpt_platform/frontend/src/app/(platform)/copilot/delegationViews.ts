import type {
  ChatDelegation,
  DelegationExpert,
  LiveDelegationStatus,
} from "./delegations";

export function formatCost(costUsd: number | null): string | null {
  return costUsd === null ? null : `$${costUsd.toFixed(2)}`;
}

export const FALLBACK_EXPERT_NAME = "Your expert";

interface KnownExpert {
  name: string;
  role: string | null;
  avatarUrl: string | null;
  color?: string | null;
}

/** Who the hand-off went to: the run's own output once it names them, else
 *  the roster entry for the id Otto passed, else the name Otto typed. */
export function resolveDelegationExpert(
  delegation: ChatDelegation,
  expertsById?: ReadonlyMap<string, KnownExpert>,
): DelegationExpert {
  if (delegation.expert) return delegation.expert;
  const id = delegation.expertId;
  const known = id ? expertsById?.get(id) : undefined;
  if (known)
    return {
      id,
      name: known.name,
      role: known.role,
      avatarUrl: known.avatarUrl,
      color: known.color ?? null,
    };
  const typed = id && !/^[0-9a-f-]{20,}$/i.test(id) ? id : null;
  return {
    id,
    name: typed ?? FALLBACK_EXPERT_NAME,
    role: null,
    avatarUrl: null,
    color: null,
  };
}

export function delegationName(
  delegation: ChatDelegation,
  expertsById?: ReadonlyMap<string, KnownExpert>,
): string {
  return resolveDelegationExpert(delegation, expertsById).name;
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

/** The task's name: the first line of what Otto asked for. */
export function delegationTitle(
  delegation: ChatDelegation,
  name = delegation.expert?.name ?? "an expert",
): string {
  const line = delegation.prompt?.split("\n")[0].trim();
  if (!line) return `Task for ${name}`;
  return line.length > 80 ? `${line.slice(0, 77)}…` : line;
}

export function threadHref(delegation: ChatDelegation): string | null {
  if (delegation.link) return delegation.link;
  return delegation.subSessionId
    ? `/copilot?sessionId=${delegation.subSessionId}`
    : null;
}
