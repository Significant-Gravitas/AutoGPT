import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { COPILOT_GATE_NODE_PREFIX } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { getAskLabel } from "../ToolChain/toolCatalog.ask";

export type ReasonKind =
  | "mode"
  | "subject"
  | "supervisor"
  | "rule"
  | "spend"
  | "content";

export type ChatRule = "allow" | "judge";

export interface ApprovalSpend {
  estimate: number;
  spent: number;
  ceiling: number;
  unit: number;
}

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
  chatRulesAllowed: ChatRule[];
  spend: ApprovalSpend | null;
  source: string | null;
  passage: string | null;
  headline: { ask: string; object: string | null };
  // Argument keys the headline already names.
  headlineKeys: string[];
}

export function isGateReview(review: PendingHumanReviewModel) {
  return review.node_exec_id.startsWith(COPILOT_GATE_NODE_PREFIX);
}

export function toApprovalItem(review: PendingHumanReviewModel): ApprovalItem {
  const payload = asRecord(review.payload);
  const toolName =
    str(payload.tool) ??
    (review.node_id ?? "").replace(COPILOT_GATE_NODE_PREFIX, "");
  const args = asRecord(payload.arguments);
  const subject = asRecord(payload.subject);
  const label = getAskLabel(toolName, args);
  const headline = label
    ? { ask: label.ask, object: label.object }
    : subject.kind === "block"
      ? { ask: "Run", object: str(subject.name) }
      : serverHeadline(
          str(payload.headline) ?? review.instructions ?? toolName,
        );
  return {
    reviewId: review.node_exec_id,
    graphExecId: review.graph_exec_id,
    toolName,
    toolCallId: str(payload.tool_call_id) ?? "",
    args,
    fields: asArray(payload.fields)
      .map(asRecord)
      .filter((f) => str(f.key))
      .map((f) => ({
        key: String(f.key),
        label: str(f.label) ?? String(f.key),
      })),
    clipped: asArray(payload.clipped).filter(
      (k): k is string => typeof k === "string",
    ),
    subject: {
      kind: str(subject.kind) ?? "tool",
      key: str(subject.key) ?? toolName,
      name: str(subject.name) ?? toolName,
      irreversible: subject.irreversible === true,
    },
    blockId: str(subject.block_id),
    reason: str(payload.reason) ?? "",
    reasonKind: (str(payload.reason_kind) as ReasonKind | null) ?? "mode",
    mode: str(payload.mode),
    chatRulesAllowed: asArray(payload.chat_rules_allowed).filter(
      (r): r is ChatRule => r === "allow" || r === "judge",
    ),
    spend: toSpend(payload.spend),
    source: str(payload.source),
    passage: str(payload.passage),
    headline,
    headlineKeys: label?.shownKeys ?? keysNamedBy(args, headline.object),
  };
}

export function approvalCardId(reviewId: string) {
  return `approval-${reviewId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

export function isHeldRead(item: ApprovalItem) {
  return item.reasonKind === "content";
}

// Said once in the queue header; per card only a reason about this call.
export function reasonLine(item: ApprovalItem): string | null {
  if (!item.reason) return null;
  switch (item.reasonKind) {
    case "supervisor":
      return `Not sure this is safe: ${item.reason}`;
    case "subject":
    case "rule":
    case "content":
      return item.reason;
    default:
      return null;
  }
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

export function visibleFieldCount(item: ApprovalItem) {
  return item.fields.filter(
    (f) => !item.headlineKeys.includes(f.key) && hasValue(item.args[f.key]),
  ).length;
}

// Nothing to read beyond the headline, so a compact line may answer it in place.
export function isBare(item: ApprovalItem) {
  return (
    visibleFieldCount(item) === 0 &&
    !reasonLine(item) &&
    !item.subject.irreversible &&
    !item.spend &&
    !isHeldRead(item)
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
      !item.spend &&
      !isHeldRead(item) &&
      (!compact || isBare(item)),
  );
}

export function approveAllLabel(count: number) {
  return count === 2 ? "Approve both" : `Approve all ${count}`;
}

export function hasValue(value: unknown) {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

function serverHeadline(text: string) {
  const match = /^(.*?) [“"](.+)[”"]$/.exec(text);
  return match
    ? { ask: match[1], object: match[2] }
    : { ask: text, object: null };
}

function keysNamedBy(args: Record<string, unknown>, object: string | null) {
  if (!object) return [];
  const prefix = object.replace(/…$/, "");
  return Object.keys(args).filter((key) => {
    const value = args[key];
    return typeof value === "string" && value.trim().startsWith(prefix);
  });
}

function toSpend(raw: unknown): ApprovalSpend | null {
  const spend = asRecord(raw);
  const numbers = ["estimate", "spent", "ceiling", "unit"].map((k) => spend[k]);
  if (!numbers.every((n) => typeof n === "number")) return null;
  const [estimate, spent, ceiling, unit] = numbers as number[];
  return { estimate, spent, ceiling, unit };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}
