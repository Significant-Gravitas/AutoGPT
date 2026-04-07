"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  icon: React.ReactNode;
  label: string;
  selected: boolean;
  onClick: () => void;
  className?: string;
}

export function SelectableCard({
  icon,
  label,
  selected,
  onClick,
  className,
}: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={cn(
        "flex h-[9rem] w-[10.375rem] shrink-0 flex-col items-center justify-center gap-3 rounded-xl border-2 bg-white px-6 py-5 transition-all hover:shadow-sm md:shrink lg:gap-2 lg:px-10 lg:py-8",
        className,
        selected
          ? "border-purple-500 bg-purple-50 shadow-sm"
          : "border-transparent",
      )}
    >
      <Text
        variant="lead"
        as="span"
        className={selected ? "text-neutral-900" : "text-purple-600"}
      >
        {icon}
      </Text>
      <Text variant="body-medium" as="span" className="whitespace-nowrap">
        {label}
      </Text>
    </button>
  );
}
