"use client";

import { FilterHorizontalIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";

interface Props {
  ariaLabelPrefix: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

export function HomeTileFilter({
  ariaLabelPrefix,
  value,
  options,
  onChange,
}: Props) {
  const activeLabel =
    options.find((option) => option.value === value)?.label ?? value;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="xs"
          className="gap-1 px-2 text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
          leadingIcon={FilterHorizontalIcon}
          aria-label={`${ariaLabelPrefix}: ${activeLabel}`}
          unmask={false}
        >
          {activeLabel}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-36">
        <DropdownMenuRadioGroup value={value} onValueChange={onChange}>
          {options.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
