import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  label: string;
  hint?: string;
  isSelected: boolean;
  onSelect: () => void;
  children: ReactNode;
}

export function OptionTile({
  label,
  hint,
  isSelected,
  onSelect,
  children,
}: Props) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      aria-label={label}
      onClick={onSelect}
      className={cn(
        "flex flex-col items-center gap-2 rounded-2xlarge border bg-white p-3 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300",
        isSelected
          ? "border-zinc-900 ring-1 ring-zinc-900"
          : "border-zinc-200 hover:border-zinc-300",
      )}
    >
      <span className="flex h-16 items-center justify-center">{children}</span>
      <span className="flex flex-col">
        <Text variant="small-medium" as="span" className="text-zinc-900">
          {label}
        </Text>
        {hint ? (
          <Text variant="small" as="span" className="text-zinc-500">
            {hint}
          </Text>
        ) : null}
      </span>
    </button>
  );
}
