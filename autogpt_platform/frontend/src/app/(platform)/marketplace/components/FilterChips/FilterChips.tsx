"use client";

import { MagnifyingGlass } from "@phosphor-icons/react";
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
  const { handleBadgeClick } = useFilterChips({
    multiSelect,
    onFilterChange,
  });

  return (
    <div className="flex h-auto min-h-8 flex-wrap items-center justify-center gap-3 lg:min-h-14 lg:justify-start">
      {badges.map((badge) => (
        <button
          key={badge}
          type="button"
          onClick={() => handleBadgeClick(badge)}
          className="group relative inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-200 via-purple-200 to-indigo-200 p-[1px] text-sm font-medium transition-all hover:scale-[1.03] md:text-lg"
        >
          <span className="inline-flex items-center gap-1.5 rounded-full bg-[rgb(246,247,248)] px-3 py-1.5 text-transparent transition-all group-hover:bg-[rgb(236,237,238)] md:gap-2 md:px-5 md:py-2">
            <MagnifyingGlass
              size={18}
              weight="regular"
              className="text-purple-300 group-hover:text-purple-400"
            />
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 bg-clip-text">
              {badge}
            </span>
          </span>
        </button>
      ))}
    </div>
  );
}
