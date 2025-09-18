"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";

interface RunListItemProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  selected?: boolean;
  onClick?: () => void;
}

export function RunSidebarCard({
  title,
  description,
  icon,
  selected,
  onClick,
}: RunListItemProps) {
  return (
    <button
      className={cn(
        "w-full rounded-large border border-slate-50/70 bg-white p-3 text-left transition-all duration-150 hover:scale-[1.01] hover:bg-slate-50/50",
        selected ? "large ring-2 ring-slate-800" : undefined,
      )}
      onClick={onClick}
    >
      <div className="flex min-w-0 items-center justify-start gap-3">
        {icon}
        <div className="flex min-w-0 flex-1 flex-col items-start justify-between">
          <Text
            variant="body-medium"
            className="block w-full truncate text-ellipsis"
          >
            {title}
          </Text>
          <Text variant="small" className="!text-zinc-500">
            {description}
          </Text>
        </div>
      </div>
    </button>
  );
}
