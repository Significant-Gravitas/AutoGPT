"use client";

import { CommandGroup, CommandList, useCommand } from "@kmenu/react";
import { useEffect } from "react";
import type {
  SearchCommandBucket,
  SearchCommandOption,
  SearchCommandOptionData,
} from "./helpers";
import { SearchCommandResultItem } from "./SearchCommandResultItem";

interface Props {
  buckets: SearchCommandBucket[];
  options: SearchCommandOption[];
  query: string;
  /** Id of the row whose action is in-flight (renders a spinner). */
  loadingItemId?: string;
}

export function SearchCommandResults({
  buckets,
  options,
  query,
  loadingItemId,
}: Props) {
  const { command, state } = useCommand<SearchCommandOptionData>();

  // A new query or a new top result moves the highlight back to the top, the
  // way Kobra's CommandMenu does; arrow navigation leaves both unchanged.
  const topId = state.filtered[0]?.id;
  useEffect(() => {
    if (topId) command?.setActiveById(topId);
  }, [command, query, topId]);

  return (
    <CommandList
      className="relative max-h-[380px] overflow-y-auto p-2"
      aria-label="Search results"
      indicatorOffsetY={-8}
    >
      {buckets.map((bucket) => {
        const bucketOptions = options.filter(
          (option) => option.group === bucket.key,
        );
        if (bucketOptions.length === 0) return null;
        return (
          <CommandGroup
            key={bucket.key}
            heading={
              <div className="relative z-1 px-2.5 pt-2 pb-1 text-xs font-medium text-muted-foreground">
                {bucket.label}
              </div>
            }
          >
            {bucketOptions.map((option) => (
              <SearchCommandResultItem
                key={option.id}
                option={option}
                query={query}
                isLoading={option.data?.item.id === loadingItemId}
              />
            ))}
          </CommandGroup>
        );
      })}
    </CommandList>
  );
}
