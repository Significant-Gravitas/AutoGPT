"use client";

import { CaretRight } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type Props = {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  showChevron?: boolean;
  rightSlot?: React.ReactNode;
  onClick: () => void;
  isActive?: boolean;
};

export function LlmMenuItem({
  title,
  subtitle,
  icon,
  showChevron,
  rightSlot,
  onClick,
  isActive,
}: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-between gap-3 rounded-md px-2 py-1.5 text-left hover:bg-zinc-100",
        isActive && "bg-zinc-100",
      )}
    >
      <div className="flex items-center gap-2">
        {icon}
        <div className="flex flex-col">
          <Text variant="body" className="text-zinc-900">
            {title}
          </Text>
          {subtitle && (
            <Text variant="small" className="text-zinc-500">
              {subtitle}
            </Text>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {rightSlot}
        {showChevron && <CaretRight className="h-4 w-4 text-zinc-800" />}
      </div>
    </button>
  );
}
