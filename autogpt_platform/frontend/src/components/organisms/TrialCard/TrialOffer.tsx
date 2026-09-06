import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatTrialPrice } from "./helpers";

interface Props {
  trial: TrialStatusResponse;
  isStarting: boolean;
  onStart: () => void;
}

export function TrialOffer({ trial, isStarting, onStart }: Props) {
  const offer = trial.offer;
  if (!offer) return null;
  return (
    <div className="flex flex-col gap-3">
      <Text variant="h4">
        Try AutoGPT{" "}
        {offer.tier === "BUSINESS" ? "Team" : offer.tier.toLowerCase()} for{" "}
        {offer.duration_days} days
      </Text>
      <Text variant="body">
        Card required. No subscription charge today. Then{" "}
        {formatTrialPrice(offer)}, plus applicable tax, unless you cancel before
        the trial ends.
      </Text>
      {trial.onboarding_credits_previously_received ||
      offer.onboarding_credit_amount > 0 ? (
        <Text variant="small">
          {trial.onboarding_credits_previously_received
            ? "You have already received your one-time onboarding credits. This trial does not add another grant."
            : `Complete onboarding for ${offer.onboarding_credit_amount} one-time onboarding credits for automations.`}
        </Text>
      ) : null}
      <Text variant="small">
        Trial usage is limited. Your billing page shows your remaining allowance
        and cancellation options.
      </Text>
      <Button
        variant="primary"
        onClick={onStart}
        loading={isStarting}
        disabled={isStarting}
      >
        Start {offer.duration_days}-day trial
      </Button>
    </div>
  );
}
