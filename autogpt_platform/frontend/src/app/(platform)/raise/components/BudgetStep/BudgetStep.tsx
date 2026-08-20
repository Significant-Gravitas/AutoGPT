"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Forward02Icon } from "@hugeicons/core-free-icons";
import { creditsToUsdLabel, kitBudgetLabel } from "../../helpers";
import { KitBudget } from "../KitStep/KitBudget";
import { DEFAULT_BUDGET_CREDITS } from "../KitStep/helpers";
import { bubbleClassFor } from "../ColorStep/helpers";
import { useBudgetStep } from "./useBudgetStep";

interface Props {
  color: string | null;
  submittedBudget: { credits: number | null } | null;
  onSubmit: (credits: number) => void;
  onSkip: () => void;
}

export function BudgetStep({
  color,
  submittedBudget,
  onSubmit,
  onSkip,
}: Props) {
  const budget = useBudgetStep({ onSubmit });

  if (submittedBudget !== null) {
    return <BudgetAnswer credits={submittedBudget.credits} color={color} />;
  }

  return (
    <div className="flex w-full flex-col items-end gap-4">
      <KitBudget
        weeklyBudget={budget.weeklyBudget}
        customCredits={budget.customCredits}
        color={color}
        onSelect={budget.selectPreset}
        onCustomChange={budget.changeCustomCredits}
      />
      <p className="max-w-[42rem] text-right text-xs text-muted-foreground">
        Skip to use the default {DEFAULT_BUDGET_CREDITS} credits (
        {creditsToUsdLabel(DEFAULT_BUDGET_CREDITS)}/week). 0 disables the weekly
        limit.
      </p>
      <div className="flex items-center gap-2">
        {budget.canSubmitCustom ? (
          <Button
            type="button"
            variant="primary"
            size="small"
            onClick={budget.submitCustom}
            className="h-[2.625rem] rounded-xl py-3"
          >
            {"That's it"}
          </Button>
        ) : null}
        <Button
          type="button"
          variant="ghost"
          size="small"
          onClick={onSkip}
          className="h-[2.625rem] rounded-xl py-3"
        >
          Skip
        </Button>
      </div>
    </div>
  );
}

function BudgetAnswer({
  credits,
  color,
}: {
  credits: number | null;
  color: string | null;
}) {
  const label = kitBudgetLabel({ weeklyBudget: credits, attachments: [] });
  const skipped = !label;
  return (
    <div
      className={cn(
        "ml-auto max-w-[80%] rounded-2xl border px-4 py-3 text-[15px] leading-relaxed text-foreground",
        skipped && "flex w-fit items-center gap-2",
        bubbleClassFor(color) ?? "border-accent bg-accent/5",
      )}
    >
      {skipped ? (
        <>
          <Icon icon={Forward02Icon} size={16} aria-hidden />
          Skipped
        </>
      ) : (
        label
      )}
    </div>
  );
}
