import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { COPILOT_GATE_NODE_PREFIX } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import {
  isIdKey,
  visibleKeys,
} from "@/components/organisms/ApprovalFields/helpers";
import { beautifyString } from "@/lib/utils";
import { asObject, str } from "../ToolChain/resultHelpers";

export type ReasonKind =
  | "mode"
  | "subject"
  | "supervisor"
  | "rule"
  | "content"
  | "spend";

// Microdollars, as the server sends them.
export interface ApprovalSpend {
  estimate: number;
  spent: number;
  ceiling: number;
  unit: number;
}

export type ChatRule = "allow" | "judge";

export interface ApprovalItem {
  reviewId: string;
  graphExecId: string;
  toolName: string;
  toolCallId: string;
  args: Record<string, unknown>;
  fields: { key: string; label: string }[];
  clipped: string[];
  subject: { kind: string; key: string; name: string; irreversible: boolean };
  blockId: string | null;
  reason: string;
  reasonKind: ReasonKind;
  mode: string | null;
  // A held read's flagged passage, which the card quotes.
  passage: string | null;
  // Over the task's spend ceiling: what this step costs and what approving adds.
  spend: ApprovalSpend | null;
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
    graphExecId: review.graph_exec_id,
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
    passage: str(payload, "passage"),
    spend: toSpend(payload.spend),
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

// A read held for carrying instructions: releasing it hands the bytes to the model.
export function isHeldRead(item: ApprovalItem) {
  return item.reasonKind === "content";
}

// Said once in the queue header; per card only a reason about this call.
export function reasonLine(item: ApprovalItem): string | null {
  if (isHeldRead(item))
    return `It contains instructions aimed at ${AUTOPILOT_NAME}, so it was held back. ${AUTOPILOT_NAME} hasn't seen it.`;
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
    return `Ask First is on, so ${AUTOPILOT_NAME} asks before it changes anything outside its workspace.`;
  return `${AUTOPILOT_NAME} asks before anything that reaches outside the platform.`;
}

export function shownFieldKeys(item: ApprovalItem) {
  // The headline names what was read; its arguments say nothing more.
  if (isHeldRead(item)) return [];
  return visibleKeys({
    keys: [...item.fields.map((f) => f.key), ...Object.keys(item.args)],
    values: item.args,
    hiddenKeys: item.headlineKeys,
    idsWhenAlone: !item.headline.object,
  });
}

// Nothing to read beyond the headline, so a compact line may answer it in place.
export function isBare(item: ApprovalItem) {
  return (
    shownFieldKeys(item).length === 0 &&
    !reasonLine(item) &&
    !item.subject.irreversible &&
    !item.spend
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
      !isHeldRead(item) &&
      !item.spend &&
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

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function toSpend(raw: unknown): ApprovalSpend | null {
  const spend = asObject(raw);
  if (!spend) return null;
  const { estimate, spent, ceiling, unit } = spend;
  if (![estimate, spent, ceiling, unit].every((n) => typeof n === "number"))
    return null;
  return { estimate, spent, ceiling, unit } as ApprovalSpend;
}
