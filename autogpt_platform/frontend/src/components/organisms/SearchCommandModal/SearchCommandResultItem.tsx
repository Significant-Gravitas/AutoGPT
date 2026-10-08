"use client";

import { Kbd } from "@/components/atoms/Kbd/Kbd";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { CommandOption, useCommand } from "@kmenu/react";
import {
  highlightMatch,
  type SearchCommandOption,
  type SearchCommandOptionData,
} from "./helpers";

interface Props {
  option: SearchCommandOption;
  query: string;
  /** Shows a trailing spinner while the row's action is in-flight. */
  isLoading?: boolean;
}

export function SearchCommandResultItem({
  option,
  query,
  isLoading = false,
}: Props) {
  const { state } = useCommand<SearchCommandOptionData>();
  const isActive = state.activeId === option.id;
  const item = option.data?.item;
  if (!item) return null;
  const RowIcon = item.icon;

  return (
    <CommandOption
      value={option}
      className="relative z-1 flex cursor-pointer items-center justify-between rounded-lg p-2.5 aria-disabled:cursor-not-allowed aria-disabled:opacity-40"
    >
      <span className="flex min-w-0 flex-1 items-center gap-2.5">
        {RowIcon ? (
          <span aria-hidden className="flex shrink-0 text-muted-foreground">
            <RowIcon className="size-[18px]" />
          </span>
        ) : null}
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-[0.9rem] text-popover-foreground">
            {highlightMatch(item.title, query).map((part, partIndex) => (
              <span
                key={`${part.text}-${partIndex}`}
                className={part.isMatch ? "font-semibold" : undefined}
              >
                {part.text}
              </span>
            ))}
          </span>
          {item.subtitle ? (
            <span className="truncate text-xs text-muted-foreground">
              {item.subtitle}
            </span>
          ) : null}
        </span>
      </span>
      {isLoading ? (
        <LoadingSpinner
          size="small"
          aria-label="Opening"
          className="shrink-0 text-muted-foreground"
        />
      ) : isActive ? (
        <Kbd aria-hidden="true" className="shrink-0">
          ↵
        </Kbd>
      ) : null}
    </CommandOption>
  );
}
