import { useState } from "react";

interface useFilterChipsProps {
  onFilterChange?: (selectedFilters: string[]) => void;
  multiSelect?: boolean;
}

export const useFilterChips = ({
  onFilterChange,
  multiSelect,
}: useFilterChipsProps) => {
  const [selectedFilters, setSelectedFilters] = useState<string[]>([]);

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

  return {
    selectedFilters,
    handleBadgeClick,
  };
};
