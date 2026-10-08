"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
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
  const [selected, setSelected] = useState(SORT_OPTIONS[0].value);
  const selectedLabel =
    SORT_OPTIONS.find((option) => option.value === selected)?.label ?? "";

  function handleValueChange(value: string) {
    setSelected(value);
    onSort(value);
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 focus-ring focus-visible:ring-offset-2">
        <Text variant="body" as="span" tone="secondary">
          Sort by
        </Text>
        <Text variant="body-medium" as="span">
          {selectedLabel}
        </Text>
        <Icon icon={ArrowDown01Icon} size={16} aria-hidden />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-50">
        <DropdownMenuRadioGroup
          value={selected}
          onValueChange={handleValueChange}
        >
          {SORT_OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
