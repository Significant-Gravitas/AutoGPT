"use client";

import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTile } from "../HomeTile/HomeTile";
import { AttentionRow } from "./components/AttentionRow";
import { NeedsYouTitle } from "./components/NeedsYouTitle";
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
        <NeedsYouTitle
          itemCount={itemCount}
          hasFilters={hasFilters}
          selectedKind={selectedKind}
          filterOptions={filterOptions}
          onSelectKind={selectKind}
          showAll={showAll}
          onToggleShowAll={() => setShowAll(!showAll)}
        />
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
