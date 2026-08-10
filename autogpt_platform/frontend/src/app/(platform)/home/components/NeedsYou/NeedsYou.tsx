"use client";

import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTileExpandButton } from "../HomeTileExpandButton/HomeTileExpandButton";
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
    showAll,
    setShowAll,
    pendingIDs,
    decide,
  } = useNeedsYou({ items: dashboard.attention });
  const itemCount = dashboard.attention.length;

  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col gap-3"
      surfaceClassName="py-4 sm:py-4"
      title={
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Icon
              icon={AlertCircleIcon}
              size={18}
              className="text-zinc-500"
              aria-hidden="true"
            />
            <Text variant="h5" className="text-zinc-950">
              Needs you
            </Text>
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            {itemCount > 0 ? (
              <span
                role="status"
                aria-label={`${itemCount} ${itemCount === 1 ? "item needs" : "items need"} your attention`}
                className="flex size-6 items-center justify-center rounded-full bg-zinc-900 text-xs font-semibold tabular-nums text-white"
              >
                {itemCount}
              </span>
            ) : null}
            {hasFilters ? (
              <HomeTileFilter
                ariaLabelPrefix="Filter interventions"
                value={selectedKind}
                options={filterOptions}
                onChange={(value) =>
                  selectKind(value as "all" | HomeAttentionItem["kind"])
                }
              />
            ) : null}
            {itemCount > 0 ? (
              <HomeTileExpandButton
                label={
                  showAll
                    ? "Show fewer attention items"
                    : "Expand attention items"
                }
                pressed={showAll}
                onClick={() => setShowAll(!showAll)}
              />
            ) : null}
          </div>
        </div>
      }
      header={
        <Text variant="large" className="max-w-xl text-pretty text-zinc-600">
          Decisions and blockers that require your input.
        </Text>
      }
    >
      {itemCount === 0 ? (
        <div className="rounded-lg bg-zinc-50 px-5 py-7 text-center ring-1 ring-inset ring-zinc-950/[0.05]">
          <Text variant="body-medium" className="text-zinc-800">
            You are all caught up
          </Text>
          <Text variant="small" className="mt-1 text-zinc-500">
            Your agents can keep moving without you.
          </Text>
        </div>
      ) : (
        <div className="-mx-4 divide-y divide-zinc-100 sm:-mx-5">
          {visibleItems.map((item) => (
            <AttentionRow
              key={item.id}
              item={item}
              isProcessing={pendingIDs.has(item.id)}
              onDecision={decide}
            />
          ))}
        </div>
      )}
    </HomeTile>
  );
}
