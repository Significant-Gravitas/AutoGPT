import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import type { SubscriptionPlansProps } from "../helpers";

type Props = Pick<
  SubscriptionPlansProps,
  "billing" | "onBillingChange" | "goalSurface"
>;

export function BillingToggle({
  billing,
  onBillingChange,
  goalSurface,
}: Props) {
  return (
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
          data-fast-goal-surface={goalSurface}
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
  );
}
