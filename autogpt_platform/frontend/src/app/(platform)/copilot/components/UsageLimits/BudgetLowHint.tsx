import { Text } from "@/components/atoms/Text/Text";
import { UsageMeters } from "@/components/organisms/UsageExperience/UsageMeters";
import { budgetLowHint } from "@/services/usageExperience/budget";
import type { UsageSnapshot } from "@/services/usageExperience/useUsageExperience";
import { cn } from "@/lib/utils";

interface Props {
  usage: UsageSnapshot;
  className?: string;
}

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
      aria-label="AutoPilot budget"
      className={cn("mx-auto mt-2 w-full space-y-2 px-3 text-left", className)}
    >
      <UsageMeters
        daily={usage.usage?.daily}
        weekly={usage.usage?.weekly}
        trialPercent={trialPercent}
      />
      <Text variant="small" className="!text-xs !text-zinc-500">
        {budgetLowHint(budget.window, percent)}
      </Text>
    </div>
  );
}
