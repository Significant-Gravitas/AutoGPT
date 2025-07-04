import { useState } from "react";
import { FilterOption } from "./SearchFilterChips";

interface useSearchFilterChipsProps {
  totalCount: number;
  agentsCount: number;
  creatorsCount: number;
  onFilterChange?: (value: string) => void;
}

export const useSearchFilterChips = ({
  totalCount,
  agentsCount,
  creatorsCount,
  onFilterChange,
}: useSearchFilterChipsProps) => {
  const [selected, setSelected] = useState("all");
  const filters: FilterOption[] = [
    { label: "All", count: totalCount, value: "all" },
    { label: "Agents", count: agentsCount, value: "agents" },
    { label: "Creators", count: creatorsCount, value: "creators" },
  ];

  const handleFilterClick = (value: string) => {
    setSelected(value);
    onFilterChange?.(value);
  };

  return {
    handleFilterClick,
    selected,
    filters,
  };
};
