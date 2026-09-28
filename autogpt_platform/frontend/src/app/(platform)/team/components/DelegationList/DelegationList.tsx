"use client";

import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useState } from "react";
import { DelegationFilterChips } from "./DelegationFilterChips";
import { DelegationListRow } from "./DelegationListRow";
import {
  type DelegationFilter,
  type DelegationFilterOption,
  filterDelegations,
  getDelegationKey,
} from "./helpers";

interface Props {
  label: string;
  delegations: DelegationSummary[];
  filters: DelegationFilterOption[];
  getMeta: (delegation: DelegationSummary) => string;
  emptyMessage: string;
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
}

export function DelegationList({
  label,
  delegations,
  filters,
  getMeta,
  emptyMessage,
  isLoading,
  isError,
  onRetry,
}: Props) {
  const [filter, setFilter] = useState<DelegationFilter>("all");
  const visible = filterDelegations(delegations, filter);

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
        onRetry={onRetry}
      />
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <DelegationFilterChips
        label={`Filter ${label.toLowerCase()}`}
        options={filters}
        value={filter}
        onChange={setFilter}
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
    </div>
  );
}
