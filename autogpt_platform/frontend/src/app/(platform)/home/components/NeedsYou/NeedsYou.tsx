"use client";

import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { HomeTileFilter } from "../HomeTileFilter/HomeTileFilter";
import { HomeTile } from "../HomeTile/HomeTile";
import { AttentionRow } from "./components/AttentionRow";
import { useNeedsYou } from "./useNeedsYou";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function NeedsYou({ dashboard, className }: Props) {
  const {
    visibleItems,
    filterOptions,
    hasFilters,
    selectedKind,
    selectKind,
    pendingIDs,
    decide,
  } = useNeedsYou({ items: dashboard.attention });
  const itemCount = dashboard.attention.length;

  return (
    <HomeTile
      className={className}
      icon={AlertCircleIcon}
      title="Needs you"
      badge={
        <span
          role="status"
          aria-label={`${itemCount} ${itemCount === 1 ? "item needs" : "items need"} your attention`}
          className="rounded-md bg-zinc-100 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-zinc-700"
        >
          {itemCount}
        </span>
      }
      meta={
        hasFilters ? (
          <HomeTileFilter
            ariaLabelPrefix="Filter interventions"
            value={selectedKind}
            options={filterOptions}
            onChange={(value) => selectKind(value as typeof selectedKind)}
          />
        ) : null
      }
    >
      <div className="divide-y divide-zinc-100">
        {visibleItems.map((item) => (
          <AttentionRow
            key={item.id}
            item={item}
            isProcessing={pendingIDs.has(item.id)}
            onDecision={decide}
          />
        ))}
      </div>
    </HomeTile>
  );
}
