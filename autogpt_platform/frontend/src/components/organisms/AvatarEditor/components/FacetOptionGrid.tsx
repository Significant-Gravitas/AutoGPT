"use client";

import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import type { AvatarConfig } from "@/components/molecules/BotAvatar/helpers";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import type { Facet } from "../helpers";

interface Props {
  facet: Facet;
  config: AvatarConfig;
  activeIndex: number;
  onPick: (index: number) => void;
  onMove: (step: number) => void;
}

export function FacetOptionGrid({
  facet,
  config,
  activeIndex,
  onPick,
  onMove,
}: Props) {
  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (isKey(event, "ArrowRight", "ArrowDown")) {
      event.preventDefault();
      onMove(1);
      return;
    }
    if (isKey(event, "ArrowLeft", "ArrowUp")) {
      event.preventDefault();
      onMove(-1);
    }
  }

  return (
    <div
      role="radiogroup"
      aria-label={facet.label}
      id={`avatar-facet-panel-${facet.id}`}
      onKeyDown={handleKeyDown}
      className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6"
    >
      {facet.options.map((option, index) => {
        const isActive = index === activeIndex;
        return (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={isActive}
            aria-label={`${option.label} — ${option.hint}`}
            tabIndex={isActive ? 0 : -1}
            data-testid={`avatar-option-${facet.id}-${option.id}`}
            onClick={() => onPick(index)}
            className={cn(
              "flex flex-col items-center gap-1 rounded-xl border-2 p-2 transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              isActive
                ? "border-zinc-900 bg-zinc-50"
                : "border-transparent hover:border-zinc-200",
            )}
          >
            <BotAvatar
              config={option.config(config)}
              size={56}
              animated={false}
              showBadge={false}
              title={option.label}
            />
            <span className="w-full truncate text-center text-xs text-zinc-600">
              {option.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
