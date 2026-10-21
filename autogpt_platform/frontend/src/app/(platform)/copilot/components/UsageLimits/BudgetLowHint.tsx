import { Text } from "@/components/atoms/Text/Text";
import { budgetLowHint } from "@/services/usageExperience/budget";
import type { UsageSnapshot } from "@/services/usageExperience/useUsageExperience";
import { cn } from "@/lib/utils";
import { BudgetRing } from "./BudgetRing";

interface Props {
  usage: UsageSnapshot;
  className?: string;
}

const WINDOW_LABEL = {
  daily: "Today",
  weekly: "This week",
  trial: "Trial allowance",
} as const;

export function BudgetLowHint({ usage, className }: Props) {
  const { budget, experience } = usage;
  if (budget.state !== "low" || !budget.window) return null;
  const trialPercent = experience.isLifetimeTrial
    ? experience.trialPercent
    : null;
  const percent =
    budget.window === "trial"
      ? (trialPercent ?? budget.percentUsed)
      : budget.percentUsed;
  if (percent === null) return null;
  return (
    <div
      aria-label="Budget running low"
      className={cn(
        "mx-auto mt-2 flex w-full items-center gap-2 px-3 text-left",
        className,
      )}
    >
      <BudgetRing label={WINDOW_LABEL[budget.window]} percent={percent} />
      <Text variant="small" className="!text-xs !text-zinc-500">
        {budgetLowHint(budget.window, percent)}
      </Text>
    </div>
  );
}
