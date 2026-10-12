import type { ApprovalItem } from "../ApprovalQueue/helpers";
import { asObject, str } from "../ToolChain/resultHelpers";
import type { ExpertIdentity } from "../../useExpertMap";

export interface HandoffExpert {
  id: string | null;
  name: string;
  role: string | null;
  avatarUrl: string | null;
  color: string | null;
}

export interface HandoffFacts {
  expert: HandoffExpert;
  title: string;
  brief: string | null;
  why: string | null;
  expectedBack: string;
  by: string | null;
}

const TITLE_MAX = 60;
const FALLBACK_NAME = "your expert";

function shortTitle(text: string | null): string | null {
  const line = text?.split("\n")[0].trim();
  if (!line) return null;
  return line.length > TITLE_MAX ? `${line.slice(0, TITLE_MAX - 1)}…` : line;
}

function expertFromPayload(
  handoff: Record<string, unknown> | null,
): HandoffExpert | null {
  const expert = handoff ? asObject(handoff.expert) : null;
  const name =
    (expert && str(expert, "name")) ?? (handoff && str(handoff, "expert_name"));
  if (!name) return null;
  return {
    id: (expert && str(expert, "id")) ?? (handoff && str(handoff, "expert_id")),
    name,
    role:
      (expert && str(expert, "role")) ??
      (handoff && str(handoff, "expert_role")),
    avatarUrl:
      (expert && str(expert, "avatar_url")) ??
      (handoff && str(handoff, "expert_avatar_url")),
    color: expert && str(expert, "color"),
  };
}

function expertFromRoster(
  id: string | null,
  expertsById: ReadonlyMap<string, ExpertIdentity>,
): HandoffExpert {
  const known = id ? expertsById.get(id) : undefined;
  if (known)
    return {
      id: known.id,
      name: known.name,
      role: known.role,
      avatarUrl: known.avatarUrl,
      color: known.color ?? null,
    };
  const typed = id && !/^[0-9a-f-]{20,}$/i.test(id) ? id : null;
  return {
    id,
    name: typed ?? FALLBACK_NAME,
    role: null,
    avatarUrl: null,
    color: null,
  };
}

/** What the approval card says about a hand-off: the review payload's
 *  `handoff` block when the server sends one, else the call's own args
 *  and the roster. */
export function readHandoff(
  item: ApprovalItem,
  payload: unknown,
  expertsById: ReadonlyMap<string, ExpertIdentity>,
): HandoffFacts {
  const handoff = asObject(asObject(payload)?.handoff);
  const args = item.args;
  const brief =
    (handoff && str(handoff, "brief", "prompt")) ?? str(args, "prompt");
  const expert =
    expertFromPayload(handoff) ??
    expertFromRoster(str(args, "expert_id"), expertsById);
  return {
    expert,
    title:
      (handoff && str(handoff, "title")) ?? shortTitle(brief) ?? "this task",
    brief,
    why:
      (handoff && str(handoff, "why", "reason")) ?? str(args, "reason", "why"),
    expectedBack:
      (handoff && str(handoff, "expected_back")) ?? "A report in this chat",
    by: handoff && str(handoff, "by", "due"),
  };
}

const MICRODOLLARS = 1_000_000;

function dollars(microdollars: number) {
  return `$${(microdollars / MICRODOLLARS).toFixed(2)}`;
}

/** The money line: what the chat has spent against its ceiling. */
export function capLine(item: ApprovalItem): string | null {
  if (!item.spend) return null;
  return `${dollars(item.spend.spent)} of ${dollars(item.spend.ceiling)}`;
}

export interface HandoffEdits {
  prompt?: string;
  expert_id?: string;
}

/** Edited args ride on the review's existing edit field; only sent when
 *  the review says it accepts edits. */
export function editedReview(item: ApprovalItem, edits: HandoffEdits) {
  if (!edits.prompt && !edits.expert_id) return {};
  return { reviewed_data: { ...item.args, ...edits } };
}
