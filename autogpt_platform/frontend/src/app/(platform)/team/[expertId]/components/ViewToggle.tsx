"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { IconSvgElement } from "@hugeicons/react";

interface Props<T extends string> {
  value: T;
  options: readonly { value: T; label: string; icon: IconSvgElement }[];
  onChange: (next: T) => void;
}

export function ViewToggle<T extends string>({
  value,
  options,
  onChange,
}: Props<T>) {
  // Sized like the search input and filter button it sits beside.
  return (
    <div className="flex h-9 items-center rounded-xl border border-input p-1">
      {options.map((option) => (
        <Button
          key={option.value}
          type="button"
          variant="toggle"
          size="icon-xs"
          className="size-7 rounded-lg"
          leadingIcon={option.icon}
          aria-label={option.label}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        />
      ))}
    </div>
  );
}
