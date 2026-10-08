"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { isComposingEvent } from "@/lib/keyboard";
import { cn } from "@/lib/utils";

type MultiToggleItem = {
  value: string;
  label: string;
  disabled?: boolean;
};

type MultiToggleProps = {
  items: MultiToggleItem[];
  selectedValues: string[];
  onChange: (selectedValues: string[]) => void;
  className?: string;
  "aria-label"?: string;
  "aria-labelledby"?: string;
};

export function MultiToggle({
  items,
  selectedValues,
  onChange,
  className,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
}: MultiToggleProps) {
  return (
    <ToggleGroup
      multiple
      value={selectedValues}
      onValueChange={(next) => onChange(next as string[])}
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      className={cn("w-full flex-wrap rounded-none", className)}
    >
      {items.map((item) => (
        <ToggleGroupItem
          key={item.value}
          value={item.value}
          disabled={item.disabled}
          variant="outline"
          size="lg"
          // Base UI presses a group item on Space keydown; an IME still
          // composing owns that key.
          onKeyDown={(event) => {
            if (isComposingEvent(event)) event.preventBaseUIHandler();
          }}
          className={cn(
            "h-9 rounded-full border-zinc-700 px-4 font-sans text-sm leading-[22px] text-foreground shadow-none hover:bg-muted hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2",
            "data-pressed:border-accent data-pressed:bg-accent/10 data-pressed:text-accent data-pressed:hover:bg-accent/15",
            item.disabled && "border-border text-zinc-200 opacity-50",
          )}
        >
          {item.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
