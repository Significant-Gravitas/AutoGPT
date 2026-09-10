"use client";

import { cn } from "@/lib/utils";
import { isKey } from "@/lib/keyboard";
import { FACETS, type FacetId } from "../helpers";

interface Props {
  value: FacetId;
  onChange: (facet: FacetId) => void;
}

export function FacetTabs({ value, onChange }: Props) {
  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!isKey(event, "ArrowLeft", "ArrowRight")) return;
    event.preventDefault();
    const step = isKey(event, "ArrowRight") ? 1 : -1;
    const index = FACETS.findIndex((facet) => facet.id === value);
    const next = (index + step + FACETS.length) % FACETS.length;
    onChange(FACETS[next].id);
  }

  return (
    <div
      role="tablist"
      aria-label="Avatar options"
      onKeyDown={handleKeyDown}
      className="inline-flex rounded-full bg-zinc-100 p-1"
    >
      {FACETS.map((facet) => {
        const isActive = facet.id === value;
        return (
          <button
            key={facet.id}
            type="button"
            role="tab"
            id={`avatar-facet-tab-${facet.id}`}
            aria-selected={isActive}
            aria-controls={`avatar-facet-panel-${facet.id}`}
            tabIndex={isActive ? 0 : -1}
            onClick={() => onChange(facet.id)}
            className={cn(
              "rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              isActive
                ? "bg-white text-zinc-900 shadow-sm"
                : "text-zinc-600 hover:text-zinc-900",
            )}
          >
            {facet.label}
          </button>
        );
      })}
    </div>
  );
}
