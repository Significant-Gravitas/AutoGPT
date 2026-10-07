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
  /** Omitted where skills are not searchable, which drops the chip. */
  skillsCount?: number;
  /** Omitted where experts are not searchable, which drops the chip. */
  expertsCount?: number;
  onFilterChange?: (value: string) => void;
}

export const SearchFilterChips: React.FC<SearchFilterChipsProps> = ({
  totalCount = 10,
  agentsCount = 8,
  creatorsCount = 2,
  skillsCount,
  expertsCount,
  onFilterChange,
}) => {
  const [selected, setSelected] = React.useState("all");

  const filters: FilterOption[] = [
    { label: "All", count: totalCount, value: "all" },
    ...(expertsCount === undefined
      ? []
      : [{ label: "Experts", count: expertsCount, value: "experts" }]),
    { label: "Agents", count: agentsCount, value: "agents" },
    ...(skillsCount === undefined
      ? []
      : [{ label: "Skills", count: skillsCount, value: "skills" }]),
    { label: "Creators", count: creatorsCount, value: "creators" },
  ];

  const handleFilterClick = (value: string) => {
    setSelected(value);
    onFilterChange?.(value);
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
              ? "cursor-not-allowed border border-zinc-200 text-zinc-300"
              : selected === filter.value
                ? "bg-zinc-800 text-white"
                : "border border-zinc-600 text-zinc-800"
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
