import { cn } from "@/lib/utils";
import { WHAT_RUNS_FILTERS, WhatRunsFilter } from "../helpers";
import { useFilterIndicator } from "./useFilterIndicator";

type Props = {
  value: WhatRunsFilter;
  onChange: (filter: WhatRunsFilter) => void;
};

export function WhatRunsFilters({ value, onChange }: Props) {
  const { listRef, indicator } = useFilterIndicator(value);

  return (
    <div
      ref={listRef}
      className="relative flex flex-wrap gap-2"
      role="group"
      aria-label="Filter"
    >
      {indicator ? (
        <span
          aria-hidden
          style={{ left: indicator.left, width: indicator.width }}
          className="ease-[cubic-bezier(0.33,1,0.68,1)] absolute top-0 h-[2.125rem] rounded-full bg-zinc-800 transition-[left,width] duration-300 motion-reduce:transition-none"
        />
      ) : null}

      {WHAT_RUNS_FILTERS.map((chip) => {
        const isActive = chip.id === value;
        return (
          <button
            key={chip.id}
            type="button"
            data-active={isActive}
            aria-pressed={isActive}
            onClick={() => onChange(chip.id)}
            className={cn(
              "relative rounded-full px-3 py-1.5 text-sm transition-colors duration-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-900",
              "active:scale-95 motion-reduce:active:scale-100",
              // The pill behind supplies the active fill; the chip only ever
              // animates its label colour, so the two never fight.
              isActive
                ? "text-white"
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
