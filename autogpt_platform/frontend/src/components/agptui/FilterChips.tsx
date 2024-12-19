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
    <div className="flex h-auto min-h-8 flex-wrap items-center justify-center gap-3 lg:min-h-14 lg:justify-start lg:gap-5">
      {badges.map((badge) => (
        <Badge
          key={badge}
          variant={selectedFilters.includes(badge) ? "secondary" : "outline"}
          className="mb-2 flex cursor-pointer items-center justify-center gap-2 rounded-full border border-black/50 px-3 py-1 dark:border-white/50 lg:mb-3 lg:gap-2.5 lg:px-6 lg:py-2"
          onClick={() => handleBadgeClick(badge)}
        >
          <div className="font-neue text-sm font-light tracking-tight text-[#474747] dark:text-[#e0e0e0] lg:text-xl lg:font-medium lg:leading-9">
            {badge}
          </div>
        </Badge>
      ))}
    </div>
  );
};
