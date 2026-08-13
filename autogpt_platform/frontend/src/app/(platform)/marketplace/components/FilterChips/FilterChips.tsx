"use client";

import { cn } from "@/lib/utils";
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
    <div className="flex flex-wrap items-center justify-center gap-2.5">
      {badges.map((badge) => {
        const isSelected = selectedFilters.includes(badge);
        return (
          <button
            key={badge}
            type="button"
            onClick={() => handleBadgeClick(badge)}
            className={cn(
              "inline-flex h-9 items-center rounded-full border px-4 text-sm font-medium transition-all duration-200",
              isSelected
                ? "border-zinc-900 bg-zinc-900 text-white shadow-[0_1px_2px_rgba(16,24,40,0.1)]"
                : "border-zinc-200 bg-white text-zinc-600 shadow-[0_1px_2px_rgba(16,24,40,0.04)] hover:-translate-y-px hover:border-zinc-300 hover:text-zinc-900",
            )}
          >
            {badge}
          </button>
        );
      })}
    </div>
  );
}
