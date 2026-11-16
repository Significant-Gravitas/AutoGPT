"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
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
          className="flex items-center gap-1.5 text-foreground-emphasis focus:outline-none"
        >
          <Text variant="large" as="span" tone="emphasis">
            Sort by
          </Text>
          <Text variant="large" as="span" tone="emphasis">
            {selected.label}
          </Text>
          <Icon aria-hidden icon={ArrowDown01Icon} className="h-4 w-4" />
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
