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
        "relative flex h-24 w-[10.375rem] shrink-0 flex-col items-center justify-center gap-2 rounded-lg border bg-surface p-4 transition-colors hover:bg-surface-muted md:shrink",
        className,
        // TODO(tokens): no fitting role for border-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour), not a border
        selected ? "border-zinc-400 bg-surface-muted" : "border-separator",
      )}
    >
      {selected && (
        <span className="absolute right-2 top-2 flex size-4 items-center justify-center rounded-full bg-action-hover">
          <Icon
            icon={Tick02Icon}
            size={10}
            className="text-action-foreground"
          />
        </span>
      )}
      <span className="flex items-center justify-center text-foreground-tertiary">
        {icon}
      </span>
      <Text variant="body-medium" as="span" className="whitespace-nowrap">
        {label}
      </Text>
    </button>
  );
}
