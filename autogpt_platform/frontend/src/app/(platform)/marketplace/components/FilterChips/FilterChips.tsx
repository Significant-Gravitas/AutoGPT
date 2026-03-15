"use client";

import { Badge } from "@/components/__legacy__/ui/badge";
import { useFilterChips } from "./useFilterChips";

interface FilterChipsProps {
  badges: string[];
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}

// Some flaws in its logic
// FRONTEND-TODO : This needs to be fixed
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
    <div className="flex h-auto min-h-8 flex-wrap items-center justify-center gap-3 lg:min-h-14 lg:justify-start lg:gap-5">
      {badges.map((badge) => (
        <Badge
          key={badge}
          variant={selectedFilters.includes(badge) ? "secondary" : "outline"}
          className="mb-2 flex cursor-pointer items-center justify-center gap-2 rounded-full border border-black/50 px-3 py-1 dark:border-white/50 lg:mb-3 lg:gap-2.5 lg:px-6 lg:py-2"
          onClick={() => handleBadgeClick(badge)}
        >
          <div className="text-sm font-light tracking-tight text-[#474747] dark:text-[#e0e0e0] lg:text-xl lg:font-medium lg:leading-9">
            {badge}
          </div>
        </Badge>
      ))}
    </div>
  );
};
