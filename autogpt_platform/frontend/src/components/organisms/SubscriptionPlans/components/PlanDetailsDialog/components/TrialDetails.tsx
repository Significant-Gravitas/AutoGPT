import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { Text } from "@/components/atoms/Text/Text";
import { formatTrialPrice } from "@/components/organisms/TrialCard/helpers";
import { formatPlanAmount } from "../../../helpers";

interface Props {
  offer: TrialOfferResponse;
}

export function TrialDetails({ offer }: Props) {
  return (
    <div className="space-y-5">
      <Text variant="body" tone="secondary">
        {offer.duration_days} days to put AutoGPT to work.
      </Text>
      <dl className="divide-y divide-zinc-200 border-y border-zinc-200">
        <div className="flex justify-between gap-4 py-4">
          <Text as="dt" variant="body">
            Today
          </Text>
          <Text as="dd" variant="lead-medium" unmask={false}>
            {formatPlanAmount(0, offer.currency)}
          </Text>
        </div>
        <div className="flex justify-between gap-4 py-4">
          <Text as="dt" variant="body">
            After {offer.duration_days} days
          </Text>
          <Text as="dd" variant="lead-medium" unmask={false}>
            {formatTrialPrice(offer)}
          </Text>
        </div>
      </dl>
      <Text variant="body" tone="secondary">
        Card required. No subscription charge today. Your paid subscription
        starts automatically after the trial, plus applicable tax, unless you
        cancel before it ends.
      </Text>
      <Text variant="body" tone="secondary">
        Trial usage is limited. Canceling ends trial access immediately. You can
        manage or cancel your plan in billing.
      </Text>
    </div>
  );
}
