"use client";

import * as React from "react";

interface FilterOption {
  label: string;
  count: number;
  value: string;
}

interface SearchFilterChipsProps {
  totalCount?: number;
  agentsCount?: number;
  creatorsCount?: number;
  onFilterChange?: (value: string) => void;
}

export const SearchFilterChips: React.FC<SearchFilterChipsProps> = ({
  totalCount = 10,
  agentsCount = 8,
  creatorsCount = 2,
  onFilterChange,
}) => {
  const [selected, setSelected] = React.useState("all");

  const filters: FilterOption[] = [
    { label: "All", count: totalCount, value: "all" },
    { label: "Agents", count: agentsCount, value: "agents" },
    { label: "Creators", count: creatorsCount, value: "creators" },
  ];

  const handleFilterClick = (value: string) => {
    setSelected(value);
    onFilterChange?.(value);
    console.log(`Filter selected: ${value}`);
  };

  return (
    <div className="flex gap-2.5">
      {filters.map((filter) => (
        <button
          key={filter.value}
          onClick={() => handleFilterClick(filter.value)}
          disabled={filter.value !== "all" && filter.count === 0}
          className={`flex items-center gap-2.5 rounded-[34px] px-5 py-2 ${
            filter.value !== "all" && filter.count === 0
              ? "cursor-not-allowed border border-neutral-200 text-neutral-300 dark:border-neutral-700 dark:text-neutral-600"
              : selected === filter.value
                ? "bg-neutral-800 text-white dark:bg-neutral-100 dark:text-neutral-900"
                : "border border-neutral-600 text-neutral-800 dark:border-neutral-400 dark:text-neutral-200"
          }`}
        >
          <span
            className={`text-base ${selected === filter.value ? "font-medium" : ""}`}
          >
            {filter.label}
          </span>
          <span
            className={`text-base ${selected === filter.value ? "font-medium" : ""}`}
          >
            {filter.count}
          </span>
        </button>
      ))}
    </div>
  );
};
