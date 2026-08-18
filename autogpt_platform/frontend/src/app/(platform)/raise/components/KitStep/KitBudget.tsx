"use client";

import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { creditsToUsdLabel } from "../../helpers";
import { bubbleClassFor } from "../ColorStep/helpers";
import { BUDGET_PRESETS } from "./helpers";

interface Props {
  weeklyBudget: number | null;
  customCredits: string;
  color: string | null;
  onSelect: (credits: number) => void;
  onCustomChange: (value: string) => void;
}

export function KitBudget({
  weeklyBudget,
  customCredits,
  color,
  onSelect,
  onCustomChange,
}: Props) {
  return (
    <div className="flex w-full max-w-[42rem] flex-col items-end gap-3">
      <div
        role="group"
        aria-label="Weekly budget"
        className="flex flex-wrap justify-end gap-2.5"
      >
        {BUDGET_PRESETS.map((preset) => {
          const selected = !customCredits && weeklyBudget === preset.credits;
          return (
            <button
              key={preset.credits}
              type="button"
              onClick={() => onSelect(preset.credits)}
              aria-pressed={selected}
              className={cn(
                "rounded-full border px-5 py-2.5 text-sm font-medium text-foreground transition-colors",
                selected
                  ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
                  : "border-border bg-background hover:border-accent hover:bg-accent/5",
              )}
            >
              {preset.label}
              {preset.credits > 0
                ? ` · ${creditsToUsdLabel(preset.credits)}`
                : ""}
            </button>
          );
        })}
      </div>
      <Input
        id="raise-kit-custom-budget"
        label="Custom weekly budget in credits"
        hideLabel
        size="small"
        inputMode="numeric"
        value={customCredits}
        onChange={(event) => onCustomChange(event.target.value)}
        placeholder="Custom credits…"
        wrapperClassName="mb-0 w-full max-w-[16rem] [&_input]:h-[2.625rem] [&_input]:py-3"
      />
    </div>
  );
}
