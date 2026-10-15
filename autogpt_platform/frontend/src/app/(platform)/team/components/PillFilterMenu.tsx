"use client";

import type { IconSvgElement } from "@hugeicons/react";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
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

export interface PillFilterOption<T extends string> {
  value: T;
  label: string;
  icon: IconSvgElement;
}

interface Props<T extends string> {
  label: string;
  value: T;
  defaultValue: T;
  options: readonly PillFilterOption<T>[];
  onChange: (next: T) => void;
}

/** A pill like the header's buttons, naming the current filter. Any choice
 *  other than the default fills it in so a narrowed list is obvious. */
export function PillFilterMenu<T extends string>({
  label,
  value,
  defaultValue,
  options,
  onChange,
}: Props<T>) {
  const current =
    options.find((option) => option.value === value) ?? options[0];
  const isActive = value !== defaultValue;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="secondary"
          size="small"
          aria-label={`${label}: ${current.label}`}
          leadingIcon={current.icon}
          rightIcon={
            <Icon
              icon={ArrowDown01Icon}
              size={14}
              className="ml-0.5 text-zinc-400 transition-transform duration-200 group-data-[state=open]:rotate-180"
            />
          }
          className={cn(
            "group shrink-0 gap-1.5 font-medium",
            isActive &&
              "border-zinc-900 bg-zinc-900 text-white hover:border-zinc-800 hover:bg-zinc-800 [&_svg]:text-white",
          )}
        >
          {current.label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[12rem]">
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(next) => onChange(next as T)}
        >
          {options.map((option) => (
            <DropdownMenuRadioItem
              key={option.value}
              value={option.value}
              className="gap-2"
            >
              <Icon icon={option.icon} size={15} className="text-zinc-500" />
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
