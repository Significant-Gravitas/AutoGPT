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
  return (
    <div className="flex h-7 items-center rounded-md border border-zinc-200 p-0.5">
      {options.map((option) => (
        <Button
          key={option.value}
          type="button"
          variant="toggle"
          size="icon-xs"
          className="size-6 rounded"
          leadingIcon={option.icon}
          aria-label={option.label}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        />
      ))}
    </div>
  );
}
