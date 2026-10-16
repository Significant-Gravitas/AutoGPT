"use client";

import { SparklesIcon } from "@hugeicons/core-free-icons";
import { CategoryChip } from "../CategoryChip/CategoryChip";
import { useFilterChips } from "./useFilterChips";

interface FilterChipsProps {
  badges: string[];
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}

export function FilterChips({
  badges,
  onFilterChange,
  multiSelect = true,
}: FilterChipsProps) {
  const { selectedFilters, handleBadgeClick } = useFilterChips({
    multiSelect,
    onFilterChange,
  });

  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {badges.map((badge) => (
        // One glyph for all of them: these are suggested searches, not
        // categories, so the category accents would misread them.
        <CategoryChip
          key={badge}
          icon={SparklesIcon}
          label={badge}
          isSelected={selectedFilters.includes(badge)}
          onClick={() => handleBadgeClick(badge)}
        />
      ))}
    </div>
  );
}
