"use client";

import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { DelegationFilterChips } from "./DelegationFilterChips";
import { DelegationListRow } from "./DelegationListRow";
import { type DelegationFilterOption, getDelegationKey } from "./helpers";
import type { useDelegationPages } from "./useDelegationPages";

export type DelegationPages = ReturnType<typeof useDelegationPages>;

interface Props {
  label: string;
  pages: DelegationPages;
  filters: DelegationFilterOption[];
  getMeta: (delegation: DelegationSummary) => string;
  emptyMessage: string;
}

export function DelegationList({
  label,
  pages,
  filters,
  getMeta,
  emptyMessage,
}: Props) {
  const { filter, delegations: visible, isLoading, isError } = pages;

  if (isLoading) {
    return (
      <div className="space-y-3" data-testid="delegation-list-loading">
        <Skeleton className="h-16 w-full rounded-lg" />
        <Skeleton className="h-16 w-full rounded-lg" />
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        context="hand-offs"
        hint="We could not load the hand-offs."
        onRetry={() => void pages.refetch()}
      />
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <DelegationFilterChips
        label={`Filter ${label.toLowerCase()}`}
        options={filters}
        value={filter}
        onChange={pages.setFilter}
      />
      {visible.length === 0 ? (
        <Text variant="body" tone="muted" className="px-1 py-6">
          {filter === "all" ? emptyMessage : "Nothing here right now."}
        </Text>
      ) : (
        <ul aria-label={label}>
          {visible.map((delegation) => (
            <li key={getDelegationKey(delegation)}>
              <DelegationListRow
                delegation={delegation}
                meta={getMeta(delegation)}
              />
            </li>
          ))}
        </ul>
      )}
      {pages.hasMore ? (
        <Button
          variant="secondary"
          size="small"
          className="self-center"
          loading={pages.isLoadingMore}
          onClick={pages.loadMore}
        >
          Show more
        </Button>
      ) : null}
    </div>
  );
}
