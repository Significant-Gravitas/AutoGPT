"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { FilterHorizontalIcon } from "@hugeicons/core-free-icons";

interface Props<T extends string> {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  defaultValue: T;
  onChange: (next: T) => void;
}

export function FilterIconMenu<T extends string>({
  label,
  value,
  options,
  defaultValue,
  onChange,
}: Props<T>) {
  const isActive = value !== defaultValue;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="icon"
          size="icon"
          aria-label={label}
          className={cn(
            "h-7 w-7 rounded-md border-zinc-200 p-0",
            isActive &&
              "border-zinc-900 bg-zinc-900 text-white hover:bg-zinc-800 hover:text-white",
          )}
        >
          <Icon icon={FilterHorizontalIcon} size={14} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[11rem]">
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(next) => onChange(next as T)}
        >
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
