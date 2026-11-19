"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { getFilterOptions } from "./helpers";

interface Props {
  totalCount: number;
  agentsCount: number;
  creatorsCount: number;
  /** Omitted where skills are not searchable, which drops the chip. */
  skillsCount?: number;
  /** Omitted where experts are not searchable, which drops the chip. */
  expertsCount?: number;
  onFilterChange: (value: string) => void;
}

export function SearchFilterChips({ onFilterChange, ...counts }: Props) {
  const [selected, setSelected] = useState("all");

  function handleFilterClick(value: string) {
    setSelected(value);
    onFilterChange(value);
  }

  return (
    <div className="flex gap-2.5">
      {getFilterOptions(counts).map((filter) => {
        const isSelected = selected === filter.value;
        const isDisabled = filter.value !== "all" && filter.count === 0;
        const variant = isSelected ? "large-medium" : "large";
        return (
          <button
            key={filter.value}
            type="button"
            aria-pressed={isSelected}
            onClick={() => handleFilterClick(filter.value)}
            disabled={isDisabled}
            className={cn(
              "flex items-center gap-2.5 rounded-full px-5 py-2",
              isDisabled
                ? "cursor-not-allowed border border-border-base text-foreground-disabled"
                : isSelected
                  ? "bg-action text-action-foreground"
                  : "border border-foreground-secondary text-foreground-emphasis",
            )}
          >
            <Text variant={variant} as="span" tone="inherit">
              {filter.label}
            </Text>
            <Text variant={variant} as="span" tone="inherit">
              {filter.count}
            </Text>
          </button>
        );
      })}
    </div>
  );
}
