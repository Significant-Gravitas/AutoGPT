import { cn } from "@/lib/utils";
import { WHAT_RUNS_FILTERS, WhatRunsFilter } from "../helpers";

interface Props {
  value: WhatRunsFilter;
  onChange: (filter: WhatRunsFilter) => void;
}

export function WhatRunsFilters({ value, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="Filter">
      {WHAT_RUNS_FILTERS.map((chip) => {
        const isActive = chip.id === value;
        return (
          <button
            key={chip.id}
            type="button"
            aria-pressed={isActive}
            onClick={() => onChange(chip.id)}
            className={cn(
              "rounded-full px-3 py-1.5 text-sm transition-colors",
              isActive
                ? "bg-zinc-800 text-white"
                : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200",
            )}
          >
            {chip.label}
          </button>
        );
      })}
    </div>
  );
}
