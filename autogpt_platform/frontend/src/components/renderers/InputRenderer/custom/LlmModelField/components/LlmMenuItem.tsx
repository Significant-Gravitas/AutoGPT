"use client";

import { CaretRight, Check } from "@phosphor-icons/react";
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
  const hasIcon = Boolean(icon);

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full px-2.5 py-1.5 text-left hover:bg-zinc-100",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {icon}
          <Text variant="body" className="text-zinc-900">
            {title}
          </Text>
        </div>
        <div className="flex items-center gap-2">
          {rightSlot}
          {isActive && <Check className="h-4 w-4 text-emerald-600" />}
          {showChevron && <CaretRight className="h-4 w-4 text-zinc-800" />}
        </div>
      </div>
      {subtitle && (
        <Text
          variant="small"
          className={cn("mt-1 text-zinc-500", hasIcon && "pl-0")}
        >
          {subtitle}
        </Text>
      )}
    </button>
  );
}
