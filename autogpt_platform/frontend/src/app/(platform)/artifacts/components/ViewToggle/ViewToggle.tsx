"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { GridViewIcon, ListViewIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { ArtifactsView } from "../../useArtifactsPage";

interface Props {
  value: ArtifactsView;
  onChange: (view: ArtifactsView) => void;
}

interface Option {
  value: ArtifactsView;
  label: string;
  icon: IconSvgElement;
}

const OPTIONS: Option[] = [
  { value: "grid", label: "Grid view", icon: GridViewIcon },
  { value: "list", label: "List view", icon: ListViewIcon },
];

export function ViewToggle({ value, onChange }: Props) {
  return (
    <div
      role="group"
      aria-label="Layout"
      className="flex items-center gap-1"
      data-testid="artifacts-view-toggle"
    >
      {OPTIONS.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-label={option.label}
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-full outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400",
              active
                ? "bg-zinc-100 text-zinc-900"
                : "text-zinc-500 hover:bg-zinc-100/70 hover:text-zinc-900",
            )}
            data-testid={`artifacts-view-${option.value}`}
          >
            <Icon icon={option.icon} size={18} />
          </button>
        );
      })}
    </div>
  );
}
