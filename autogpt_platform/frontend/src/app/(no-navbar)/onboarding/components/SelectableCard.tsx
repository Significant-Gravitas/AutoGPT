"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
        "relative flex h-24 w-[10.375rem] shrink-0 flex-col items-center justify-center gap-2 rounded-lg border bg-white p-4 transition-colors hover:bg-zinc-50 md:shrink",
        className,
        selected ? "border-zinc-400 bg-zinc-50" : "border-zinc-100",
      )}
    >
      {selected && (
        <span className="absolute right-2 top-2 flex size-4 items-center justify-center rounded-full bg-zinc-900">
          <Icon icon={Tick02Icon} size={10} className="text-white" />
        </span>
      )}
      <span className="flex items-center justify-center text-zinc-500">
        {icon}
      </span>
      <Text variant="body-medium" as="span" className="whitespace-nowrap">
        {label}
      </Text>
    </button>
  );
}
