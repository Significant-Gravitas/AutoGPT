import { useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { groupByExpert, undecidedHeldCalls } from "./helpers";
import { useAttentionDecisions } from "./useAttentionDecisions";
import { useHeldReview } from "./useHeldReview";

interface Args {
  items: HomeAttentionItem[];
}

type AttentionFilter = "all" | HomeAttentionItem["kind"];

const FILTER_LABELS: Partial<Record<AttentionFilter, string>> = {
  all: "All",
  approval: "Approvals",
  setup: "Setup",
  paused: "Paused",
  credits: "Credits",
  question: "Questions",
};

export function useNeedsYou({ items }: Args) {
  const { pendingIDs, decide } = useAttentionDecisions();
  const held = useHeldReview({ items });
  const [activeKind, setActiveKind] = useState<AttentionFilter>("all");
  const [confirmRejectAll, setConfirmRejectAll] = useState(false);
  const filterKinds = Array.from(new Set(items.map((item) => item.kind)));
  const selectedKind: AttentionFilter =
    activeKind !== "all" && filterKinds.includes(activeKind)
      ? activeKind
      : "all";
  const visibleRows =
    selectedKind === "all"
      ? held.rows
      : held.rows.filter((row) => row.item.kind === selectedKind);

  function selectKind(kind: AttentionFilter) {
    setActiveKind(kind);
  }

  return {
    groups: groupByExpert(visibleRows),
    rejectable: undecidedHeldCalls(visibleRows),
    confirmRejectAll,
    setConfirmRejectAll,
    visibleRows,
    pendingCount: held.pendingCount,
    filterOptions: (["all", ...filterKinds] as AttentionFilter[]).map(
      (value) => ({ value, label: FILTER_LABELS[value] ?? value }),
    ),
    hasFilters: filterKinds.length > 1,
    selectedKind,
    selectKind,
    pendingIDs,
    decide,
    held,
  };
}
