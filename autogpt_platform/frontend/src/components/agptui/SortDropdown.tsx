"use client";

import * as React from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownIcon } from "@radix-ui/react-icons";

const sortOptions: SortOption[] = [
  { label: "Trending", value: "trending" },
  { label: "Most Recent", value: "recent" },
  { label: "Most Popular", value: "popular" },
  { label: "Highest Rated", value: "rating" },
  { label: "Most Runs", value: "runs" },
];

interface SortOption {
  label: string;
  value: string;
}

export const SortDropdown: React.FC = () => {
  const [selected, setSelected] = React.useState(sortOptions[0]);

  const handleSelect = (option: SortOption) => {
    setSelected(option);
    console.log(`Sorting by: ${option.label} (${option.value})`);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="flex items-center gap-1.5 focus:outline-none">
        <span className="text-neutral-800 text-base font-medium">Sort by</span>
        <span className="text-neutral-800 text-base font-medium">{selected.label}</span>
        <ChevronDownIcon className="h-4 w-4 text-neutral-800" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[200px] bg-white rounded-lg shadow-lg">
        {sortOptions.map((option) => (
          <DropdownMenuItem
            key={option.value}
            className={`px-4 py-2 text-base cursor-pointer hover:bg-neutral-100 ${
              selected.value === option.value ? "text-neutral-800 font-medium" : "text-neutral-600"
            }`}
            onClick={() => handleSelect(option)}
          >
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};