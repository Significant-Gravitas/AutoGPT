"use client";

import { useState } from "react";
import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";

export type BriefingFilter = "all" | HomeBriefingOutcome["status"];

const FILTER_LABELS: Partial<Record<BriefingFilter, string>> = {
  all: "All",
  completed: "Completed",
  failed: "Failed",
};

export function useMorningBriefing({
  outcomes,
}: {
  outcomes: HomeBriefingOutcome[];
}) {
  const [activeFilter, setActiveFilter] = useState<BriefingFilter>("all");
  const filterStatuses = Array.from(
    new Set(outcomes.map((outcome) => outcome.status)),
  );
  const selectedFilter: BriefingFilter =
    activeFilter !== "all" && filterStatuses.includes(activeFilter)
      ? activeFilter
      : "all";
  const visibleOutcomes =
    selectedFilter === "all"
      ? outcomes
      : outcomes.filter((outcome) => outcome.status === selectedFilter);

  return {
    filterOptions: (["all", ...filterStatuses] as BriefingFilter[]).map(
      (value) => ({ value, label: FILTER_LABELS[value] ?? value }),
    ),
    hasFilters: filterStatuses.length > 1,
    selectedFilter,
    selectFilter: setActiveFilter,
    visibleOutcomes,
  };
}
