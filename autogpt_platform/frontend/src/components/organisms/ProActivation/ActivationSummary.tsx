import type { ActivationTerms } from "@/app/api/__generated__/models/activationTerms";
import { formatAmount, intervalLabel } from "@/services/pro-activation/helpers";
import {
  billingCadence,
  invoiceAdjustment,
  shortDate,
} from "@/services/pro-activation/reviewDetails";

interface Props {
  terms: ActivationTerms;
}

export function ActivationSummary({ terms }: Props) {
  const plan = terms.plan === "MAX" ? "Max" : "Pro";
  const cadence = billingCadence(terms);
  const adjustment = invoiceAdjustment(terms);
  const period =
    terms.period_starts_at && terms.next_payment_at
      ? `${plan}, ${shortDate(terms.period_starts_at)} to ${shortDate(terms.next_payment_at)}`
      : `${plan} subscription`;
  return (
    <div className="overflow-hidden rounded-xlarge border border-border-base">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 border-b border-border-base bg-surface-muted px-4 py-3.5">
        <p className="text-sm text-foreground-tertiary">
          <span className="text-base font-semibold text-foreground-strong">
            {plan}
          </span>{" "}
          ·{" "}
          <span className="capitalize">
            {cadence.charAt(0).toUpperCase() + cadence.slice(1)}
          </span>
        </p>
        <p className="text-sm tabular-nums text-foreground-secondary">
          {formatAmount(terms.renewal_unit_amount, terms.currency)} /{" "}
          {intervalLabel(
            terms.billing_interval,
            terms.billing_interval_count ?? 1,
          )}
        </p>
      </div>
      <dl className="space-y-3.5 p-4 text-sm text-foreground-secondary">
        <div className="flex items-start justify-between gap-4">
          <dt>{period}</dt>
          <dd className="shrink-0 tabular-nums">
            {formatAmount(terms.renewal_unit_amount, terms.currency)}
          </dd>
        </div>
        {!!adjustment && (
          <div className="flex items-start justify-between gap-4">
            <dt>Discounts, credits &amp; adjustments</dt>
            <dd className="shrink-0 tabular-nums">
              {formatAmount(adjustment, terms.currency)}
            </dd>
          </div>
        )}
        <div className="flex items-start justify-between gap-4">
          <dt>
            {terms.renewal_tax?.price_tax_behavior === "inclusive"
              ? "Tax (included)"
              : "Tax"}
          </dt>
          <dd className="shrink-0 tabular-nums">
            {terms.tax_amount == null
              ? "Included in total"
              : formatAmount(terms.tax_amount, terms.currency)}
          </dd>
        </div>
        <div className="flex items-baseline justify-between gap-4 border-t border-border-base pt-3.5 text-base font-semibold text-foreground-strong">
          <dt>Due today</dt>
          <dd className="tabular-nums">
            {formatAmount(terms.amount_due, terms.currency)}
          </dd>
        </div>
      </dl>
    </div>
  );
}
