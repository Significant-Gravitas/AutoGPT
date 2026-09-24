import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { COPILOT_GATE_NODE_PREFIX } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import {
  isIdKey,
  type Reference,
  visibleKeys,
} from "@/components/organisms/ApprovalFields/helpers";
import { beautifyString } from "@/lib/utils";
import { asObject, str } from "../ToolChain/resultHelpers";

export type ReasonKind = "mode" | "subject" | "supervisor" | "rule";

export type ChatRule = "allow" | "judge";

export interface ApprovalItem {
  reviewId: string;
  // Which list the answer refreshes: a chat's queue, or a run's.
  scope: Pick<PendingHumanReviewModel, "graph_exec_id" | "session_id">;
  toolName: string;
  toolCallId: string;
  args: Record<string, unknown>;
  fields: { key: string; label: string }[];
  references: Reference[];
  // Ids per argument before the server clipped it.
  referenceTotals: Record<string, number>;
  clipped: string[];
  subject: { kind: string; key: string; name: string; irreversible: boolean };
  blockId: string | null;
  reason: string;
  reasonKind: ReasonKind;
  mode: string | null;
  chatRulesAllowed: ChatRule[];
  headline: { ask: string; object: string | null };
  // The argument the headline already names.
  headlineKeys: string[];
}

export function isGateReview(review: PendingHumanReviewModel) {
  return review.node_exec_id.startsWith(COPILOT_GATE_NODE_PREFIX);
}

export function toApprovalItem(review: PendingHumanReviewModel): ApprovalItem {
  const payload = asObject(review.payload) ?? {};
  const toolName =
    str(payload, "tool") ??
    (review.node_id ?? "").replace(COPILOT_GATE_NODE_PREFIX, "");
  const subject = asObject(payload.subject) ?? {};
  const headline = asObject(payload.headline) ?? {};
  const objectKey = str(headline, "object_key");
  return {
    reviewId: review.node_exec_id,
    scope: {
      graph_exec_id: review.graph_exec_id,
      session_id: review.session_id,
    },
    toolName,
    toolCallId: str(payload, "tool_call_id") ?? "",
    args: asObject(payload.arguments) ?? {},
    fields: asArray(payload.fields)
      .map((f) => asObject(f) ?? {})
      .filter((f) => str(f, "key"))
      .map((f) => ({
        key: String(f.key),
        label: str(f, "label") ?? String(f.key),
      })),
    references: asArray(payload.references).flatMap(toReference),
    referenceTotals: toTotals(payload.reference_totals),
    clipped: asArray(payload.clipped).filter(
      (k): k is string => typeof k === "string",
    ),
    subject: {
      kind: str(subject, "kind") ?? "tool",
      key: str(subject, "key") ?? toolName,
      name: str(subject, "name") ?? toolName,
      irreversible: subject.irreversible === true,
    },
    blockId: str(subject, "block_id"),
    reason: str(payload, "reason") ?? "",
    reasonKind: (str(payload, "reason_kind") as ReasonKind | null) ?? "mode",
    mode: str(payload, "mode"),
    chatRulesAllowed: asArray(payload.chat_rules_allowed).filter(
      (r): r is ChatRule => r === "allow" || r === "judge",
    ),
    headline:
      subject.kind === "block"
        ? { ask: "Run", object: str(subject, "name") }
        : {
            ask: str(headline, "ask") ?? fallbackAsk(toolName),
            object: str(headline, "object"),
          },
    headlineKeys: objectKey ? [objectKey] : [],
  };
}

// A row the server wrote no headline for still names its tool.
export function fallbackAsk(toolName: string) {
  return `Run ${beautifyString(toolName.replace(/^run_/, "")).toLowerCase()}`;
}

export function approvalCardId(reviewId: string) {
  return `approval-${reviewId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

// Said once in the queue header; per card only a reason about this call.
export function reasonLine(item: ApprovalItem): string | null {
  if (!item.reason) return null;
  if (item.reasonKind === "supervisor")
    return `Not sure this is safe: ${item.reason}`;
  if (item.reasonKind === "subject" || item.reasonKind === "rule")
    return item.reason;
  return null;
}

export function modeLabel(mode: string | null) {
  if (mode === "ask_first") return "Ask First";
  if (mode === "unsupervised") return "Unsupervised";
  if (mode === "auto") return "Auto";
  return null;
}

export function modeLine(mode: string | null) {
  if (mode === "ask_first")
    return `Ask First is on, so ${AUTOPILOT_NAME} asks before he changes anything outside his workspace.`;
  return `${AUTOPILOT_NAME} asks before anything that reaches outside the platform.`;
}

export function shownFieldKeys(item: ApprovalItem) {
  return visibleKeys({
    keys: [...item.fields.map((f) => f.key), ...Object.keys(item.args)],
    values: item.args,
    hiddenKeys: item.headlineKeys,
    idsWhenAlone: !item.headline.object,
    references: item.references,
  });
}

// Nothing to read beyond the headline, so a compact line may answer it in place.
export function isBare(item: ApprovalItem) {
  return (
    shownFieldKeys(item).length === 0 &&
    !reasonLine(item) &&
    !item.subject.irreversible
  );
}

export const MAX_APPROVE_ALL = 5;

export function canApproveAll(items: ApprovalItem[], compact: boolean) {
  if (items.length < 2 || items.length > MAX_APPROVE_ALL) return false;
  const key = items[0].subject.key;
  return items.every(
    (item) =>
      item.subject.key === key &&
      !item.subject.irreversible &&
      !isIdOnly(item) &&
      (!compact || isBare(item)),
  );
}

export function approveAllLabel(count: number) {
  return count === 2 ? "Approve both" : `Approve all ${count}`;
}

// Told apart only by ids, so a set of them cannot be approved as one.
function isIdOnly(item: ApprovalItem) {
  const keys = shownFieldKeys(item);
  return !item.headline.object && keys.length > 0 && keys.every(isIdKey);
}

function toReference(value: unknown): Reference[] {
  const ref = asObject(value) ?? {};
  const key = str(ref, "key");
  const id = str(ref, "id");
  if (!key || !id) return [];
  const name = str(ref, "name");
  return [
    {
      key,
      id,
      entity: str(ref, "entity") ?? "",
      name,
      // A link is only ever built for an id that resolved.
      href: name ? safeHref(str(ref, "href")) : null,
      summary: name ? str(ref, "summary") : null,
    },
  ];
}

// Only an in-app path: the payload is stored data, never a place to send the user.
function safeHref(href: string | null) {
  return href && href.startsWith("/") && !href.startsWith("//") ? href : null;
}

function toTotals(value: unknown): Record<string, number> {
  return Object.fromEntries(
    Object.entries(asObject(value) ?? {}).filter(
      (entry): entry is [string, number] => typeof entry[1] === "number",
    ),
  );
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
