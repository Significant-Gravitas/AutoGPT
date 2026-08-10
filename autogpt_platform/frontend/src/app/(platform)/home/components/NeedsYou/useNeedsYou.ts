import { useState } from "react";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { useAttentionDecisions } from "./useAttentionDecisions";

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
};

export function useNeedsYou({ items }: Args) {
  const { pendingIDs, decide } = useAttentionDecisions();
  const [showAll, setShowAll] = useState(false);
  const [activeKind, setActiveKind] = useState<AttentionFilter>("all");
  const filterKinds = Array.from(new Set(items.map((item) => item.kind)));
  const selectedKind: AttentionFilter =
    activeKind !== "all" && filterKinds.includes(activeKind)
      ? activeKind
      : "all";
  const filteredItems =
    selectedKind === "all"
      ? items
      : items.filter((item) => item.kind === selectedKind);
  const visibleItems = showAll ? filteredItems : filteredItems.slice(0, 3);

  function selectKind(kind: AttentionFilter) {
    setActiveKind(kind);
    setShowAll(false);
  }

  return {
    filteredItems,
    visibleItems,
    filterOptions: (["all", ...filterKinds] as AttentionFilter[]).map(
      (value) => ({ value, label: FILTER_LABELS[value] ?? value }),
    ),
    hasFilters: filterKinds.length > 1,
    selectedKind,
    selectKind,
    showAll,
    setShowAll,
    pendingIDs,
    decide,
  };
}
