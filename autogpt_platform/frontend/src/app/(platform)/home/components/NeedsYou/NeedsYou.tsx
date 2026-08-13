"use client";

import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
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
        />
      }
      header={
        <Text variant="large" className="max-w-xl text-pretty text-zinc-600">
          Decisions and blockers that require your input.
        </Text>
      }
    >
      {itemCount === 0 ? (
        <HomeTileEmpty
          icon={CheckmarkCircle02Icon}
          title="You are all caught up"
          description="Your agents can keep moving without you."
        />
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
