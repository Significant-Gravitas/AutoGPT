"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
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
    <div className="flex items-center rounded-xl border border-zinc-200 p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-label={option.label}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "flex h-7 w-8 items-center justify-center rounded-lg text-zinc-500 transition-colors hover:text-zinc-800",
            value === option.value && "bg-zinc-100 text-zinc-900",
          )}
        >
          <Icon icon={option.icon} size={15} />
        </button>
      ))}
    </div>
  );
}
