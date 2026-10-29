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
      className="inline-flex rounded-full border border-border-base bg-surface-hover p-0.5"
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
                ? "bg-surface text-foreground-strong shadow-sm"
                : "bg-transparent text-foreground-tertiary hover:text-foreground-body",
              disabled && "cursor-not-allowed opacity-60",
            )}
          >
            {cycle === "monthly" ? (
              "Monthly"
            ) : (
              <>
                Yearly{" "}
                <span
                  // oxlint-disable-next-line design-system/no-raw-colors -- the green-to-teal "Save 15%" gradient has no role
                  className="ml-1 bg-gradient-to-r from-green-500 via-success to-teal-500 bg-clip-text text-caption-sm font-semibold leading-inherit text-transparent"
                >
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
