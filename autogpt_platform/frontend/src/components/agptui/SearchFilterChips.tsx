"use client";

import * as React from "react";
import { Button } from "../ui/button";
import AutogptButton from "./AutogptButton";

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
        <AutogptButton
          key={filter.value}
          variant={selected === filter.value ? "default" : "outline"}
          onClick={() => handleFilterClick(filter.value)}
        >
          <span
            className={`mr-2 font-sans text-base ${selected === filter.value ? "font-medium" : "font-normal"}`}
          >
            {filter.label}
          </span>
          <span
            className={`font-sans text-base ${selected === filter.value ? "font-medium" : "font-normal"}`}
          >
            {filter.count}
          </span>
        </AutogptButton>
      ))}
    </div>
  );
};
