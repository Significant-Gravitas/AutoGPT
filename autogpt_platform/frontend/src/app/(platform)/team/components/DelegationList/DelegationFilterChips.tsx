import { cn } from "@/lib/utils";
import type { DelegationFilter, DelegationFilterOption } from "./helpers";

interface Props {
  label: string;
  options: DelegationFilterOption[];
  value: DelegationFilter;
  onChange: (value: DelegationFilter) => void;
}

export function DelegationFilterChips({
  label,
  options,
  value,
  onChange,
}: Props) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap gap-1">
      {options.map((option) => {
        const isActive = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={isActive}
            onClick={() => onChange(option.value)}
            className={cn(
              "h-8 rounded-full px-3 font-sans text-sm leading-[22px] transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400",
              isActive
                ? "bg-zinc-100 font-medium text-black"
                : "text-zinc-600 hover:bg-zinc-50",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
