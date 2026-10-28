"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { motion } from "framer-motion";

export type Tone =
  | "accent"
  | "warning"
  | "info"
  | "success"
  | "error"
  | "neutral";

export interface Reason {
  label: string;
  icon: IconSvgElement;
  tone: Tone;
}

const TONES: Record<Tone, string> = {
  accent: "bg-accent-surface text-accent-strong",
  warning: "bg-warning-surface text-warning-foreground",
  info: "bg-info-surface text-info-foreground",
  success: "bg-success-surface text-success-foreground",
  error: "bg-error-surface text-error-foreground",
  neutral: "bg-surface-hover text-foreground-secondary",
};

interface Props {
  reasons: Reason[];
  label: string;
  selected: string | null;
  onSelect: (label: string) => void;
  columns?: 2 | 3 | 4;
  layout?: "row" | "stack";
  size?: "default" | "compact";
}

const COLUMNS = { 2: "grid-cols-2", 3: "grid-cols-3", 4: "grid-cols-4" };

export function ReasonTiles({
  reasons,
  label,
  selected,
  onSelect,
  columns = 2,
  layout = "row",
  size = "default",
}: Props) {
  const isCompact = size === "compact";
  const spanLast = layout === "row" && reasons.length % columns !== 0;

  return (
    <div
      role="group"
      aria-label={label}
      className={cn("grid gap-2 p-px", COLUMNS[columns])}
    >
      {reasons.map((reason, index) => {
        const isSelected = selected === reason.label;
        return (
          <motion.button
            key={reason.label}
            type="button"
            aria-pressed={isSelected}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSelect(reason.label)}
            className={cn(
              "relative flex rounded-xl border bg-surface text-left transition-[border-color,box-shadow,background-color] duration-150",
              "border-separator hover:border-border-base hover:shadow-sm",
              isSelected && "border-action shadow-sm ring-1 ring-action",
              layout === "row"
                ? cn("items-center", isCompact ? "gap-2 p-2" : "gap-3 p-3")
                : "flex-col items-center gap-2 px-2 py-3 text-center",
              spanLast && index === reasons.length - 1 && "col-span-full",
            )}
          >
            <span
              className={cn(
                "flex shrink-0 items-center justify-center rounded-lg",
                isCompact ? "size-7" : "size-9",
                TONES[reason.tone],
              )}
            >
              <Icon icon={reason.icon} size={isCompact ? 15 : 18} />
            </span>
            <span
              className={cn(
                "min-w-0 break-words font-medium leading-5 text-foreground-emphasis",
                isCompact ? "text-caption leading-4" : "text-body",
              )}
            >
              {reason.label}
            </span>
            <span
              className={cn(
                "absolute right-1.5 top-1.5 flex size-4 items-center justify-center rounded-full bg-action text-action-foreground transition-[opacity,transform] duration-150",
                isSelected ? "scale-100 opacity-100" : "scale-50 opacity-0",
              )}
            >
              <Icon icon={Tick02Icon} size={10} strokeWidth={3} />
            </span>
          </motion.button>
        );
      })}
    </div>
  );
}
