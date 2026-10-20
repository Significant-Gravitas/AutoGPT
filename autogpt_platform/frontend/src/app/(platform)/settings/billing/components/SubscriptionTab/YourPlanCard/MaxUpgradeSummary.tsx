import type { SubscriptionUpgradePreview } from "@/app/api/__generated__/models/subscriptionUpgradePreview";
import { Text } from "@/components/atoms/Text/Text";
import { formatAmount } from "@/services/pro-activation/helpers";

interface Props {
  preview: SubscriptionUpgradePreview;
}

export function formatUpgradeAmount(amount: number, currency: string) {
  return formatAmount(amount, currency);
}

function dateLabel(timestamp: number) {
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function MaxUpgradeSummary({ preview }: Props) {
  const amount = (cents: number) =>
    formatUpgradeAmount(cents, preview.currency);
  const unit = preview.billing_cycle === "yearly" ? "year" : "month";
  const recurringTax = preview.recurring_tax_included
    ? "including tax"
    : "plus applicable tax";
  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-200">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-zinc-200 bg-zinc-50 px-5 py-4">
        <Text variant="body-medium">
          Max{" "}
          <span className="font-normal text-zinc-500">
            · {preview.billing_cycle === "yearly" ? "Yearly" : "Monthly"}
          </span>
        </Text>
        <Text variant="small" tone="secondary">
          {amount(preview.recurring_amount)} / {unit}
        </Text>
      </div>
      <div className="space-y-3.5 p-5">
        {preview.lines.map((line, index) => (
          <SummaryRow
            key={index}
            label={
              line.proration
                ? line.amount < 0
                  ? "Unused Pro time"
                  : `Max, ${dateLabel(line.period_start)} to ${dateLabel(line.period_end)}`
                : line.description
            }
            value={amount(line.amount)}
          />
        ))}
        <SummaryRow
          label={preview.tax_included ? "Tax (included)" : "Tax"}
          value={
            preview.tax_amount == null
              ? "See invoice"
              : amount(preview.tax_amount)
          }
        />
        {preview.adjustment_amount !== 0 && (
          <SummaryRow
            label="Other invoice adjustments"
            value={amount(preview.adjustment_amount)}
          />
        )}
        <div className="border-t border-zinc-200 pt-4">
          <SummaryRow
            label="Due today"
            value={amount(preview.amount_due)}
            emphasized
          />
        </div>
        <Text variant="small" tone="secondary" className="pt-2 leading-5">
          Next renewal: {dateLabel(preview.period_end)}. Max’s regular price is{" "}
          {amount(preview.recurring_amount)} {recurringTax} per {unit}, before
          any applicable discounts or account credits.
        </Text>
      </div>
    </div>
  );
}

function SummaryRow({
  label,
  value,
  emphasized = false,
}: {
  label: string;
  value: string;
  emphasized?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <Text
        variant={emphasized ? "body-medium" : "small"}
        className={emphasized ? "text-zinc-900" : "text-zinc-600"}
      >
        {label}
      </Text>
      <Text
        variant={emphasized ? "body-medium" : "small"}
        className={`shrink-0 tabular-nums ${emphasized ? "text-zinc-900" : "text-zinc-600"}`}
      >
        {value}
      </Text>
    </div>
  );
}
