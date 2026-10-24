"use client";

import type { MemoryFact } from "@/app/api/__generated__/models/memoryFact";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatWhen } from "../helpers";

interface Props {
  facts: MemoryFact[];
  total: number | null;
  isLoading: boolean;
  isExpanded: boolean;
  canShowAll: boolean;
  hasMore: boolean;
  isAllLoaded: boolean;
  isLoadingMore: boolean;
  onShowAll: () => void;
  onShowFewer: () => void;
  onLoadMore: () => void;
  onForget: (uuid: string) => void;
  onForgetTopic: () => void;
  onForgetEverything: () => void;
}

export function RecentMemoriesCard({
  facts,
  total,
  isLoading,
  isExpanded,
  canShowAll,
  hasMore,
  isAllLoaded,
  isLoadingMore,
  onShowAll,
  onShowFewer,
  onLoadMore,
  onForget,
  onForgetTopic,
  onForgetEverything,
}: Props) {
  return (
    <div className="flex flex-col rounded-[18px] border border-border-base bg-surface px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
      <div className="flex items-baseline justify-between gap-3">
        <Text variant="body-medium" as="span" className="text-textBlack">
          {isExpanded ? "All memories" : "Recent memories"}
        </Text>
        {total !== null && total > 0 ? (
          <Text
            variant="small"
            as="span"
            className="shrink-0 text-foreground-tertiary"
          >
            {total === 1 ? "1 memory" : `${total} memories`}
          </Text>
        ) : null}
      </div>
      <Text variant="small" as="span" className="text-foreground-tertiary">
        {isExpanded
          ? "Everything this memory holds, newest first. Forget any of them."
          : "The latest things this memory has learned. Forget any of them."}
      </Text>

      <div className="mt-2 flex flex-col divide-y divide-separator">
        {isLoading ? (
          <div className="flex flex-col gap-2 py-3">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-5 w-2/3" />
            <Skeleton className="h-5 w-4/5" />
          </div>
        ) : facts.length === 0 ? (
          <Text
            variant="small"
            as="p"
            className="py-3 text-foreground-tertiary"
          >
            Nothing remembered yet. Memories show up here as you chat.
          </Text>
        ) : (
          facts.map((fact) => (
            <MemoryRow key={fact.uuid} fact={fact} onForget={onForget} />
          ))
        )}
      </div>

      {canShowAll || hasMore || isExpanded ? (
        <div className="flex flex-wrap gap-2 pt-1">
          {canShowAll ? (
            <Button variant="ghost" size="small" onClick={onShowAll}>
              {total !== null && isAllLoaded ? `Show all ${total}` : "Show all"}
            </Button>
          ) : null}
          {hasMore ? (
            <Button
              variant="ghost"
              size="small"
              loading={isLoadingMore}
              onClick={onLoadMore}
            >
              Show more
            </Button>
          ) : null}
          {isExpanded ? (
            <Button variant="ghost" size="small" onClick={onShowFewer}>
              Show fewer
            </Button>
          ) : null}
        </div>
      ) : null}

      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="secondary" size="small" onClick={onForgetTopic}>
          Forget a topic…
        </Button>
        {total !== null && total > 0 ? (
          <Button variant="secondary" size="small" onClick={onForgetEverything}>
            Forget everything…
          </Button>
        ) : null}
      </div>
    </div>
  );
}

interface MemoryRowProps {
  fact: MemoryFact;
  onForget: (uuid: string) => void;
}

function MemoryRow({ fact, onForget }: MemoryRowProps) {
  return (
    <div className="flex items-center justify-between gap-3 py-2.5">
      <Text variant="small" as="span" className="min-w-0 flex-1 text-textBlack">
        {fact.fact || `${fact.source} → ${fact.target}`}
      </Text>
      <Text
        variant="small"
        as="span"
        className="hidden shrink-0 text-foreground-subtle sm:inline"
      >
        {formatWhen(fact.created_at)}
      </Text>
      <Button
        variant="ghost"
        size="small"
        className="h-7 !min-w-0 shrink-0 px-2 text-foreground-secondary"
        onClick={() => onForget(fact.uuid)}
      >
        Forget
      </Button>
    </div>
  );
}
