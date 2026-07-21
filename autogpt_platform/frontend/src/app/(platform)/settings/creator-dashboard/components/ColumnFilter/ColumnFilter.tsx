"use client";

import { FunnelIcon } from "@phosphor-icons/react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";

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
            "ease-[cubic-bezier(0.16,1,0.3,1)] transition-[transform,background-color,color] duration-150",
            "active:scale-[0.92] motion-reduce:transition-none motion-reduce:active:scale-100",
            active
              ? "bg-violet-100 text-violet-700 hover:bg-violet-200"
              : "text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700",
          )}
        >
          <FunnelIcon size={12} weight={active ? "fill" : "bold"} />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align={align}
        sideOffset={6}
        className={cn(
          "w-64 p-3 will-change-transform",
          "origin-[var(--radix-popover-content-transform-origin)]",
          "data-[state=closed]:duration-150 data-[state=open]:duration-200",
          "data-[state=open]:ease-[cubic-bezier(0.16,1,0.3,1)]",
          "data-[state=closed]:ease-[cubic-bezier(0.4,0,1,1)]",
          "motion-reduce:!animate-none motion-reduce:!duration-100",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </PopoverContent>
    </Popover>
  );
}
