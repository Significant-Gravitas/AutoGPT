"use client";

import { Button } from "@/components/atoms/Button/Button";
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
        {/* Sized and bordered like the small SearchInput it sits beside,
            with no fill until a filter is active. */}
        <Button
          type="button"
          variant="outline"
          size="icon-sm"
          aria-label={label}
          leadingIcon={FilterHorizontalIcon}
          className={cn(
            "size-9 rounded-xl border-input bg-transparent text-zinc-600 shadow-none hover:border-input hover:bg-zinc-50",
            isActive &&
              "border-zinc-900 bg-zinc-900 text-white hover:border-zinc-800 hover:bg-zinc-800 hover:text-white",
          )}
        />
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
