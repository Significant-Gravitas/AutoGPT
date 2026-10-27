"use client";

import { cn } from "@/lib/utils";
import React, { useId } from "react";
import { isKey } from "@/lib/keyboard";

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
  const groupId = useId();

  function handleToggle(value: string) {
    if (selectedValues.includes(value)) {
      onChange(selectedValues.filter((v) => v !== value));
    } else {
      onChange([...selectedValues, value]);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent, value: string) {
    if (isKey(event, " ", "Enter")) {
      event.preventDefault();
      handleToggle(value);
    }
  }

  return (
    <div
      role="group"
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      data-slot="multi-toggle"
      className={cn("flex flex-wrap gap-2", className)}
    >
      {items.map((item) => {
        const isSelected = selectedValues.includes(item.value);
        const itemId = `${groupId}-${item.value}`;

        return (
          <button
            key={item.value}
            id={itemId}
            type="button"
            role="checkbox"
            aria-checked={isSelected}
            disabled={item.disabled}
            onClick={() => handleToggle(item.value)}
            onKeyDown={(e) => handleKeyDown(e, item.value)}
            data-slot="multi-toggle-item"
            // TODO(tokens): no role for purple-50 (#F1EBFE) or purple-100
            // (#EFE8FE); zinc-700 and zinc-200 have only text and surface
            // roles, not a border or text one.
            className={cn(
              // Base button styles similar to outline variant
              "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-hover focus-visible:ring-offset-2",
              "disabled:pointer-events-none disabled:opacity-50",
              "rounded-full border font-sans",
              "h-[2.25rem] px-4 py-2 text-body leading-5.5",
              // Default outline styles
              "border-zinc-700 bg-transparent text-foreground-base hover:bg-surface-hover",
              // Selected styles with purple-600
              isSelected &&
                "border-brand-hover bg-purple-50 text-brand-hover hover:bg-purple-100",
              // Disabled styles
              item.disabled && "border-border-base text-zinc-200 opacity-50",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
