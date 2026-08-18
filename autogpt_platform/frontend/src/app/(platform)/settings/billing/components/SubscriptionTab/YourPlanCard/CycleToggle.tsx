"use client";

import { cn } from "@/lib/utils";

interface Props {
  value: "monthly" | "yearly";
  onChange: (cycle: "monthly" | "yearly") => void;
  disabled?: boolean;
}

const CYCLES = ["monthly", "yearly"] as const;

export function CycleToggle({ value, onChange, disabled }: Props) {
  return (
    <div
      role="radiogroup"
      aria-label="Billing cycle"
      className="inline-flex rounded-full border border-[#d8d8d8] bg-zinc-100 p-[3px]"
    >
      {CYCLES.map((cycle) => {
        const selected = value === cycle;
        return (
          <button
            key={cycle}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(cycle)}
            className={cn(
              "rounded-full border-none px-3 py-1 text-xs font-medium transition-all",
              selected
                ? "bg-white text-zinc-900 shadow-sm"
                : "bg-transparent text-zinc-500 hover:text-zinc-700",
              disabled && "cursor-not-allowed opacity-60",
            )}
          >
            {cycle === "monthly" ? (
              "Monthly"
            ) : (
              <>
                Yearly{" "}
                <span className="ml-1 bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 bg-clip-text text-[11px] font-semibold text-transparent">
                  Save 15%
                </span>
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}
