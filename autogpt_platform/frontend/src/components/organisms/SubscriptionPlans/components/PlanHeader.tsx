import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { SubscriptionPlansProps } from "../helpers";

export function PlanHeader({
  billing,
  onBillingChange,
}: Pick<SubscriptionPlansProps, "billing" | "onBillingChange">) {
  return (
    <header className="mb-5 flex flex-col items-center text-center">
      <AutoGPTLogo hideText className="relative right-5 mb-2 h-8 w-20" />
      <Text variant="h3" as="h1" className="leading-9">
        Choose the plan that&apos;s right for{" "}
        <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
          you
        </span>
      </Text>
      <Text variant="body" tone="muted" className="mt-1">
        Upgrade, downgrade, or change plans anytime. All plans include core
        features to get you started.
      </Text>
      <div
        role="group"
        aria-label="Billing period"
        className="mt-3 flex rounded-full border border-zinc-200 bg-zinc-100 p-1"
      >
        {(["monthly", "yearly"] as const).map((cycle) => (
          <Button
            key={cycle}
            type="button"
            variant="ghost"
            size="xs"
            aria-pressed={billing === cycle}
            onClick={() => onBillingChange(cycle)}
            data-fast-goal="paywall_billing_toggle"
            data-fast-goal-cycle={cycle}
            data-fast-goal-surface="onboarding_paywall"
            className={cn(
              "rounded-full px-4 text-xs",
              billing === cycle
                ? "bg-white text-zinc-900 shadow-sm hover:bg-white"
                : "text-zinc-500",
            )}
          >
            {cycle === "monthly" ? "Monthly billing" : "Yearly billing"}
            {cycle === "yearly" && (
              <span className="text-xs text-emerald-600">Save 15%</span>
            )}
          </Button>
        ))}
      </div>
    </header>
  );
}
