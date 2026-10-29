"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
// oxlint-disable-next-line design-system/hugeicons-only -- the chevron keeps its glyph until the icon pass (P11)
import { ChevronDownIcon } from "@radix-ui/react-icons";
import { useState } from "react";

const SORT_OPTIONS = [
  { label: "Most Runs", value: "runs" },
  { label: "Highest Rated", value: "rating" },
  { label: "Name (A-Z)", value: "name" },
  { label: "Recently Updated", value: "updated_at" },
];

interface Props {
  onSort: (sortValue: string) => void;
}

export function SortDropdown({ onSort }: Props) {
  const [selected, setSelected] = useState(SORT_OPTIONS[0]);

  function handleSelect(option: (typeof SORT_OPTIONS)[number]) {
    setSelected(option);
    onSort(option.value);
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex items-center gap-1.5 focus:outline-none"
        >
          <Text variant="large" as="span" tone="emphasis">
            Sort by
          </Text>
          <Text variant="large" as="span" tone="emphasis">
            {selected.label}
          </Text>
          <ChevronDownIcon className="h-4 w-4 text-foreground-emphasis" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[200px]">
        {SORT_OPTIONS.map((option) => {
          const isSelected = selected.value === option.value;
          return (
            <DropdownMenuItem
              key={option.value}
              // oxlint-disable-next-line design-system/no-restyle -- keeps the sort menu's 16px row padding; the molecule's 8px rows would be a layout change
              className="px-4 py-2"
              onClick={() => handleSelect(option)}
            >
              <Text
                variant={isSelected ? "large-medium" : "large"}
                as="span"
                tone={isSelected ? "emphasis" : "secondary"}
              >
                {option.label}
              </Text>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
