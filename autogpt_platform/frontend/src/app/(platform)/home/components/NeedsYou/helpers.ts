import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";

/** A held hand-off to a teammate: its review payload carries the card. */
export function isHandoffApproval(item: HomeAttentionItem): boolean {
  const payload = item.review?.payload;
  if (item.kind !== "approval" || !payload || typeof payload !== "object")
    return false;
  if (Array.isArray(payload)) return false;
  const handoff = (payload as Record<string, unknown>).handoff;
  return typeof handoff === "object" && handoff !== null;
}

const KIND_TAGS: Partial<
  Record<HomeAttentionItem["kind"], { label: string; className: string }>
> = {
  question: {
    label: "Question",
    className: "bg-amber-50 text-amber-800 ring-amber-500/20",
  },
  approval: {
    label: "Approval",
    className: "bg-violet-50 text-violet-700 ring-violet-600/15",
  },
};

export function getKindTag(kind: HomeAttentionItem["kind"]) {
  return KIND_TAGS[kind] ?? null;
}
