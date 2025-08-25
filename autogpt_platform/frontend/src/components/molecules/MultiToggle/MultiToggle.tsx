"use client";

import { cn } from "@/lib/utils";
import React, { useId } from "react";

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
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      handleToggle(value);
    }
  }

  return (
    <div
      role="group"
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
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
            className={cn(
              // Base button styles similar to outline variant
              "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-600 focus-visible:ring-offset-2",
              "disabled:pointer-events-none disabled:opacity-50",
              "rounded-full border font-sans",
              "h-[2.25rem] px-4 py-2 text-sm leading-[22px]",
              // Default outline styles
              "border-zinc-700 bg-transparent text-black hover:bg-zinc-100",
              // Selected styles with purple-600
              isSelected &&
                "border-purple-600 bg-purple-50 text-purple-600 hover:bg-purple-100",
              // Disabled styles
              item.disabled && "border-zinc-200 text-zinc-200 opacity-50",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
