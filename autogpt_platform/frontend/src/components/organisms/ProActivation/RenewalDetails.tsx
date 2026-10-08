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
  const tax = terms.renewal_tax;
  return (
    <div className="space-y-2">
      <Text variant="small" className="font-medium text-zinc-800">
        Renewal details
      </Text>
      <Text variant="small" className="text-zinc-600">
        {terms.renewal_terms}
      </Text>
      {terms.renewal_discounts?.map((discount, index) => (
        <Text key={index} variant="small" className="text-zinc-600">
          {discountText(discount, terms.currency)}
        </Text>
      ))}
      {tax?.automatic && (
        <Text variant="small" className="text-zinc-600">
          Tax is calculated automatically for each invoice.
        </Text>
      )}
      <Text variant="small" className="text-zinc-600">
        {tax?.price_tax_behavior === "inclusive"
          ? "The plan price includes tax."
          : tax?.price_tax_behavior === "exclusive"
            ? "Applicable tax is added to the plan price."
            : "The final renewal total depends on applicable tax and offer terms."}
      </Text>
      {tax?.rates?.map((rate, index) => (
        <Text key={index} variant="small" className="text-zinc-600">
          {rate.display_name}: {rate.percentage}%{" "}
          {rate.inclusive ? "included" : "additional"}
          {[rate.country, rate.state].filter(Boolean).length
            ? ` · ${[rate.country, rate.state].filter(Boolean).join(", ")}`
            : ""}
        </Text>
      ))}
    </div>
  );
}
