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
  /** `body-md` inside a profile tab, whose copy reads at 15px. */
  textSize?: "body" | "body-md";
}

/** A pill like the header's buttons, naming the current filter. Any choice
 *  other than the default fills it in so a narrowed list is obvious. */
export function PillFilterMenu<T extends string>({
  label,
  value,
  defaultValue,
  options,
  onChange,
  textSize = "body",
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
          textSize={textSize}
          aria-label={`${label}: ${current.label}`}
          leadingIcon={current.icon}
          rightIcon={
            <Icon
              icon={ArrowDown01Icon}
              size={14}
              className="ml-0.5 text-foreground-subtle transition-transform duration-200 group-data-[state=open]:rotate-180"
            />
          }
          className={cn(
            "group shrink-0 gap-1.5 font-medium",
            isActive &&
              "border-action-hover bg-action-hover text-action-foreground hover:border-action hover:bg-action [&_svg]:text-action-foreground",
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
              <Icon
                icon={option.icon}
                size={15}
                className="text-foreground-tertiary"
              />
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
