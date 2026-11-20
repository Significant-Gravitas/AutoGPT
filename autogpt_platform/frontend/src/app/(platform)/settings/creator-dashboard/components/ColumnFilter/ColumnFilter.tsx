"use client";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { FunnelIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  active: boolean;
  label: string;
  align?: "start" | "center" | "end";
  children: ReactNode;
}

export function ColumnFilter({
  active,
  label,
  align = "start",
  children,
}: Props) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`Filter ${label}`}
          className={cn(
            "inline-flex h-6 w-6 items-center justify-center rounded-full",
            "transition-[transform,background-color,color] duration-150",
            "active:scale-[0.92] motion-reduce:transition-none motion-reduce:active:scale-100",
            active
              ? "bg-brand/10 text-accent-strong hover:bg-brand/20"
              : "text-foreground-subtle hover:bg-surface-hover hover:text-foreground-body",
          )}
        >
          <Icon icon={FunnelIcon} size={12} />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align={align}
        sideOffset={6}
        // oxlint-disable-next-line design-system/no-restyle -- the column filter is a compact popover with its own open and close timing
        className="w-64 origin-[var(--radix-popover-content-transform-origin)] p-3 will-change-transform data-[state=closed]:duration-150 data-[state=open]:duration-200 motion-reduce:!animate-none motion-reduce:!duration-100"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </PopoverContent>
    </Popover>
  );
}
