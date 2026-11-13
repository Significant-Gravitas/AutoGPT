import { useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { useAttentionDecisions } from "./useAttentionDecisions";
import { useHeldReview } from "./useHeldReview";
import { useReviewCarousel } from "./useReviewCarousel";

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
  const carousel = useReviewCarousel({ rows: held.rows, held });
  const [activeKind, setActiveKind] = useState<AttentionFilter>("all");
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
    carousel,
  };
}
