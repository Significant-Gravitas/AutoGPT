"use client";

import * as React from "react";
import { Badge } from "@/components/ui/badge";

interface FilterChipsProps {
  badges: string[];
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}
/** FilterChips is a component that allows the user to select filters from a list of badges. It is used on the Marketplace home page */
export const FilterChips: React.FC<FilterChipsProps> = ({
  badges,
  onFilterChange,
  multiSelect = true,
}) => {
  const [selectedFilters, setSelectedFilters] = React.useState<string[]>([]);

  const handleBadgeClick = (badge: string) => {
    setSelectedFilters((prevFilters) => {
      let newFilters;
      if (multiSelect) {
        newFilters = prevFilters.includes(badge)
          ? prevFilters.filter((filter) => filter !== badge)
          : [...prevFilters, badge];
      } else {
        newFilters = prevFilters.includes(badge) ? [] : [badge];
      }

      if (onFilterChange) {
        onFilterChange(newFilters);
      }

      return newFilters;
    });
  };

  return (
    <div className="flex flex-wrap items-center justify-center gap-3">
      {badges.map((badge) => (
        <Badge
          data-testid="filter-chip"
          key={badge}
          variant={selectedFilters.includes(badge) ? "secondary" : "outline"}
          className="rounded-[2rem] border border-neutral-600 px-5 py-3 hover:cursor-pointer hover:bg-neutral-200"
          onClick={() => handleBadgeClick(badge)}
        >
          <p className="font-sans text-base font-normal text-neutral-800 md:text-xl">
            {badge}
          </p>
        </Badge>
      ))}
    </div>
  );
};
