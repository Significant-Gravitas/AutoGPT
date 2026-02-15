"use client";

import { FilterChip } from "@/components/atoms/FilterChip/FilterChip";
import { useFilterChips } from "./useFilterChips";

interface FilterChipsProps {
  badges: string[];
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}

export const FilterChips = ({
  badges,
  onFilterChange,
  multiSelect = true,
}: FilterChipsProps) => {
  const { selectedFilters, handleBadgeClick } = useFilterChips({
    multiSelect,
    onFilterChange,
  });

  return (
    <div
      className="flex h-auto min-h-8 flex-wrap items-center justify-center gap-3 lg:min-h-14 lg:justify-start lg:gap-5"
      role="group"
      aria-label="Filter options"
    >
      {badges.map((badge) => (
        <FilterChip
          key={badge}
          label={badge}
          selected={selectedFilters.includes(badge)}
          onClick={() => handleBadgeClick(badge)}
          size="lg"
          className="mb-2 lg:mb-3"
        />
      ))}
    </div>
  );
};
