"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";

interface Props {
  totalCount: number;
  agentsCount: number;
  creatorsCount: number;
  /** Omitted where skills are not searchable, which drops the chip. */
  skillsCount?: number;
  /** Omitted where experts are not searchable, which drops the chip. */
  expertsCount?: number;
  onFilterChange?: (value: string) => void;
}

export function SearchFilterChips({
  totalCount,
  agentsCount,
  creatorsCount,
  skillsCount,
  expertsCount,
  onFilterChange,
}: Props) {
  const [selected, setSelected] = useState("all");

  const filters = [
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

  function handleFilterClick(value: string) {
    setSelected(value);
    onFilterChange?.(value);
  }

  return (
    <div className="flex flex-wrap gap-2.5">
      {filters.map((filter) => {
        const isSelected = selected === filter.value;
        return (
          <Button
            key={filter.value}
            variant={isSelected ? "primary" : "outline"}
            size="md"
            aria-pressed={isSelected}
            disabled={filter.value !== "all" && filter.count === 0}
            onClick={() => handleFilterClick(filter.value)}
          >
            {filter.label}
            <span className={isSelected ? undefined : "text-muted-foreground"}>
              {filter.count}
            </span>
          </Button>
        );
      })}
    </div>
  );
}
