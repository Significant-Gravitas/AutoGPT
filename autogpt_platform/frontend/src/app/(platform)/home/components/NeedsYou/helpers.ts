import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import {
  type ApprovalItem,
  canApproveAll,
  isBare,
  isGateReview,
  isHeldRead,
  toApprovalItem,
} from "@/app/(platform)/copilot/components/ApprovalQueue/helpers";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import type { AttentionListRow } from "./useHeldReview";

// A passage this short is shown whole on the row, so it can be released from there.
export const PASSAGE_FITS = 180;

// A held AutoPilot call the gate wrote a headline for; older rows keep the generic row.
export function isHeldCall(item: HomeAttentionItem) {
  return (
    item.kind === "approval" &&
    !!item.review &&
    isGateReview(item.review) &&
    !!item.headline
  );
}

// The row shows everything the card would, so it may be approved without opening it.
export function isInformed(item: ApprovalItem) {
  if (item.spend || item.subject.irreversible) return false;
  if (isHeldRead(item))
    return item.unjudged || (item.passage ?? "").length <= PASSAGE_FITS;
  return isBare(item);
}

export function receiptText(item: ApprovalItem, approved: boolean) {
  if (isHeldRead(item))
    return approved
      ? `Released · ${AUTOPILOT_NAME} is reading it`
      : `Kept out · ${AUTOPILOT_NAME} was told`;
  return approved
    ? `Approved · ${AUTOPILOT_NAME} is on it`
    : `Rejected · ${AUTOPILOT_NAME} was told`;
}

// "3h", "40m": narrow enough for a column of twenty rows.
export function shortAge(createdAt: Date | string, now = new Date()) {
  const minutes = Math.max(
    1,
    Math.floor((now.getTime() - new Date(createdAt).getTime()) / 60_000),
  );
  const [value, unit] =
    minutes < 60
      ? [minutes, "minute"]
      : minutes < 24 * 60
        ? [Math.floor(minutes / 60), "hour"]
        : [Math.floor(minutes / (24 * 60)), "day"];
  return new Intl.NumberFormat(undefined, {
    style: "unit",
    unit,
    unitDisplay: "narrow",
  }).format(value);
}

export function headlineButtonId(itemID: string) {
  return `held-${itemID.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

export interface AttentionGroup {
  key: string;
  // Null for Otto's own chats and for rows that belong to no Expert.
  expert: HomeAttentionItem["expert"];
  rows: AttentionListRow[];
}

// One Expert's rows together, groups in the order their first row was drawn,
// so a call arriving on the poll joins the end of its group.
export function groupByExpert(rows: AttentionListRow[]): AttentionGroup[] {
  const groups = new Map<string, AttentionGroup>();
  for (const row of rows) {
    const { item } = row;
    const key = item.expert?.id ?? (isHeldCall(item) ? "otto" : item.id);
    const group = groups.get(key) ?? {
      key,
      expert: item.expert ?? null,
      rows: [],
    };
    group.rows.push(row);
    groups.set(key, group);
  }
  return [...groups.values()];
}

export function undecidedHeldCalls(rows: AttentionListRow[]) {
  return rows
    .filter((row) => !row.receipt && isHeldCall(row.item))
    .map((row) => row.item);
}

// The chat's rule for a set, and every call in it already shown whole on its row.
export function canApproveGroup(items: HomeAttentionItem[]) {
  const approvals = items.map((item) => toApprovalItem(item.review!));
  return canApproveAll(approvals, false) && approvals.every(isInformed);
}
