import type { ActivationTerms as Terms } from "@/app/api/__generated__/models/activationTerms";
import { Text } from "@/components/atoms/Text/Text";
import { formatAmount, intervalLabel } from "@/services/pro-activation/helpers";
import { RenewalDetails } from "./RenewalDetails";

interface Props {
  terms: Terms;
}

export function ActivationTerms({ terms }: Props) {
  return (
    <div className="space-y-5">
      <div className="rounded-xlarge border border-violet-100 bg-violet-50/50 p-5">
        <div className="flex items-baseline justify-between gap-4">
          <Text variant="h3" className="font-semibold text-zinc-900">
            Pro
          </Text>
          <Text variant="body" className="text-zinc-600">
            {formatAmount(terms.renewal_unit_amount, terms.currency)} /{" "}
            {intervalLabel(
              terms.billing_interval,
              terms.billing_interval_count ?? 1,
            )}
          </Text>
        </div>
        <Text variant="small" className="mt-2 text-zinc-600">
          Fresh daily and weekly usage when your paid plan activates.
        </Text>
      </div>
      <div className="flex items-baseline justify-between gap-4 border-b border-zinc-100 pb-5">
        <div>
          <Text variant="body" className="font-medium">
            Due on confirmation
          </Text>
          <Text variant="small" className="text-zinc-500">
            Includes applicable tax, discounts, and credits
          </Text>
        </div>
        <Text variant="h3" className="whitespace-nowrap font-semibold">
          {formatAmount(terms.amount_due, terms.currency)}
        </Text>
      </div>
      <RenewalDetails terms={terms} />
      <Text variant="small" className="text-zinc-500">
        Quote valid until {new Date(terms.expires_at).toLocaleString()}. Your
        free trial ends when you confirm.
      </Text>
    </div>
  );
}
