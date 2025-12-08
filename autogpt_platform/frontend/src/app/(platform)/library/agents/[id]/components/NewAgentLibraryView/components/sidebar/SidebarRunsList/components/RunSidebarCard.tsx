"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import React from "react";

interface RunListItemProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  statusBadge?: React.ReactNode;
  selected?: boolean;
  onClick?: () => void;
}

export function RunSidebarCard({
  title,
  description,
  icon,
  statusBadge,
  selected,
  onClick,
}: RunListItemProps) {
  return (
    <button
      className={cn(
        "w-full rounded-large border border-zinc-200 bg-white p-3 text-left ring-1 ring-transparent transition-all duration-150 hover:scale-[1.01] hover:bg-slate-50/50",
        selected ? "border-slate-800 ring-slate-800" : undefined,
      )}
      onClick={onClick}
    >
      <div className="flex min-w-0 items-center justify-start gap-3">
        {icon}
        <div className="flex min-w-0 flex-1 flex-col items-start justify-between gap-0">
          <div className="flex w-full items-center justify-between">
            <Text
              variant="body-medium"
              className="block truncate text-ellipsis"
            >
              {title}
            </Text>
            {statusBadge}
          </div>
          <Text variant="body" className="leading-tight !text-zinc-500">
            {description}
          </Text>
        </div>
      </div>
    </button>
  );
}
