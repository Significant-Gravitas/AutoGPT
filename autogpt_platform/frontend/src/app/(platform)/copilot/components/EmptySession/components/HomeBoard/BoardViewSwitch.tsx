"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { KanbanIcon, ListViewIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { BoardView } from "./useBoardView";

interface Props {
  value: BoardView;
  onChange: (view: BoardView) => void;
}

const OPTIONS: { value: BoardView; label: string; icon: IconSvgElement }[] = [
  { value: "board", label: "Board view", icon: KanbanIcon },
  { value: "list", label: "List view", icon: ListViewIcon },
];

/** Shows the tasks as a board or as a list. */
export function BoardViewSwitch({ value, onChange }: Props) {
  return (
    <div
      role="group"
      aria-label="Layout"
      className="inline-flex h-8 shrink-0 items-center gap-0.5 rounded-lg bg-surface p-0.5 shadow-(--inbox-shadow-card)"
    >
      {OPTIONS.map((option) => {
        const isActive = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-label={option.label}
            title={option.label}
            aria-pressed={isActive}
            onClick={() => onChange(option.value)}
            className={cn(
              "inline-flex size-7 items-center justify-center rounded-md transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox motion-reduce:transition-none",
              isActive
                ? "bg-surface-wash text-foreground-ink"
                : "text-foreground-faint hover:text-foreground-ink",
            )}
          >
            <Icon icon={option.icon} size={15} tone="inherit" />
          </button>
        );
      })}
    </div>
  );
}
