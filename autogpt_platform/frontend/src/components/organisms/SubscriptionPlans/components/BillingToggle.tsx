import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import type { SubscriptionPlansProps } from "../helpers";

type Props = Pick<SubscriptionPlansProps, "billing" | "onBillingChange">;

export function BillingToggle({ billing, onBillingChange }: Props) {
  return (
    <div
      role="group"
      aria-label="Billing period"
      data-slot="billing-toggle"
      className="mt-3 flex rounded-full border border-border-base bg-surface-hover p-1"
    >
      {(["monthly", "yearly"] as const).map((cycle) => (
        <Button
          key={cycle}
          type="button"
          variant="ghost"
          size="xs"
          aria-pressed={billing === cycle}
          onClick={() => onBillingChange(cycle)}
          className={cn(
            "rounded-full px-4 text-xs",
            billing === cycle
              ? "bg-surface text-foreground-strong shadow-sm hover:bg-surface"
              : "text-foreground-tertiary",
          )}
        >
          {cycle === "monthly" ? "Monthly billing" : "Yearly billing"}
          {cycle === "yearly" && (
            <span
              data-slot="billing-toggle-savings"
              // TODO(tokens): no role for emerald-600; success is green-600.
              className="text-xs text-emerald-600"
            >
              Save 15%
            </span>
          )}
        </Button>
      ))}
    </div>
  );
}
