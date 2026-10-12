"use client";

import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTileFilter } from "../HomeTileFilter/HomeTileFilter";
import { HomeTile } from "../HomeTile/HomeTile";
import { AttentionRow } from "./components/AttentionRow";
import { HeldCall } from "./components/HeldCall";
import { HeldReviewDialog } from "./components/HeldReviewDialog";
import { isHeldCall } from "./helpers";
import { useNeedsYou } from "./useNeedsYou";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function NeedsYou({ dashboard, className }: Props) {
  const {
    visibleRows,
    pendingCount,
    filterOptions,
    hasFilters,
    selectedKind,
    selectKind,
    pendingIDs,
    decide,
    held,
    carousel,
  } = useNeedsYou({ items: dashboard.attention });

  return (
    <HomeTile
      className={className}
      icon={AlertCircleIcon}
      title="Needs you"
      badge={
        <Text
          variant="small-medium"
          as="span"
          tone="secondary"
          role="status"
          aria-label={`${pendingCount} ${pendingCount === 1 ? "item needs" : "items need"} your attention`}
          className="rounded-md bg-zinc-100 px-1.5 py-0.5 tabular-nums"
        >
          {pendingCount}
        </Text>
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
        {visibleRows.map(({ item, receipt }) =>
          isHeldCall(item) ? (
            <HeldCall
              key={item.id}
              item={item}
              receipt={receipt}
              status={held.statusOf(item.id)}
              failed={held.hasFailed(item.id)}
              onOpen={() => carousel.openAt(item.id)}
              onDecide={(approved) => held.decide([item], approved)}
            />
          ) : (
            <AttentionRow
              key={item.id}
              item={item}
              isProcessing={pendingIDs.has(item.id)}
              onDecision={decide}
            />
          ),
        )}
      </div>
      <span className="sr-only" aria-live="polite">
        {held.announcement}
      </span>
      <HeldReviewDialog carousel={carousel} held={held} />
    </HomeTile>
  );
}
