import type { ActivationTerms } from "@/app/api/__generated__/models/activationTerms";
import type { RenewalDiscount } from "@/app/api/__generated__/models/renewalDiscount";
import { Text } from "@/components/atoms/Text/Text";
import { formatAmount } from "@/services/pro-activation/helpers";

function discountText(discount: RenewalDiscount, currency: string) {
  const amount =
    discount.percent_off != null
      ? `${discount.percent_off}% off`
      : discount.amount_off != null
        ? `${formatAmount(discount.amount_off, discount.currency ?? currency)} off`
        : "Offer discount";
  const duration =
    discount.duration === "once"
      ? "the first invoice only"
      : discount.duration === "forever"
        ? "every renewal"
        : discount.duration_in_months
          ? `for ${discount.duration_in_months} months`
          : "for the offer period";
  return `${amount} ${duration}${discount.ends_at ? ` · ends ${new Date(discount.ends_at * 1000).toLocaleDateString()}` : ""}`;
}

interface Props {
  terms: ActivationTerms;
}

export function RenewalDetails({ terms }: Props) {
  return (
    <div data-slot="renewal-details" className="space-y-2">
      {terms.renewal_discounts?.map((discount, index) => (
        <Text key={index} variant="small" tone="secondary">
          {discountText(discount, terms.currency)}
        </Text>
      ))}
      <details
        data-slot="renewal-details-terms"
        className="group text-sm text-foreground-tertiary"
      >
        <summary
          data-slot="renewal-details-terms-toggle"
          // TODO(tokens): no role for the violet-500 focus outline.
          className="w-fit cursor-pointer rounded-small underline decoration-border-strong underline-offset-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-violet-500"
        >
          Full renewal terms
        </summary>
        <Text variant="small" tone="secondary" className="mt-3 leading-relaxed">
          {terms.renewal_terms}
        </Text>
        {terms.renewal_tax?.rates?.map((rate, index) => (
          <Text key={index} variant="small" tone="secondary" className="mt-2">
            {rate.display_name}: {rate.percentage}%{" "}
            {rate.inclusive ? "included" : "additional"}
            {[rate.country, rate.state].filter(Boolean).length
              ? ` · ${[rate.country, rate.state].filter(Boolean).join(", ")}`
              : ""}
          </Text>
        ))}
      </details>
    </div>
  );
}
