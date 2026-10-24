"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import React from "react";

interface Props {
  title: string;
  description?: React.ReactNode;
  descriptionTitle?: string;
  icon?: React.ReactNode;
  selected?: boolean;
  onClick?: () => void;
  actions?: React.ReactNode;
  testID?: string;
}

export function SidebarItemCard({
  title,
  description,
  descriptionTitle,
  icon,
  selected,
  onClick,
  actions,
  testID,
}: Props) {
  return (
    <div
      className={cn(
        // TODO(tokens): no exact role for hover:bg-slate-50/50 (#F7F8F9)
        "w-full cursor-pointer rounded-large border border-border-base bg-surface p-3 text-left ring-1 ring-transparent transition-all duration-150 hover:scale-[1.01] hover:bg-slate-50/50",
        // TODO(tokens): no exact role for border-slate-800 (#475263), ring-slate-800 (#475263)
        selected ? "border-slate-800 ring-slate-800" : undefined,
      )}
      onClick={onClick}
      data-testid={testID}
    >
      <div className="flex min-w-0 items-center justify-start gap-3">
        {icon}
        <div className="flex min-w-0 flex-1 flex-col items-start justify-between gap-0">
          <Text
            variant="body-medium"
            className="block w-full truncate text-ellipsis"
          >
            {title}
          </Text>
          <Text
            variant="body"
            className="leading-tight !text-foreground-tertiary"
            title={descriptionTitle}
          >
            {description}
          </Text>
        </div>
        {actions ? (
          <div onClick={(e) => e.stopPropagation()}>{actions}</div>
        ) : null}
      </div>
    </div>
  );
}
