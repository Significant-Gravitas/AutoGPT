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
      <Text variant="small">
        Trial usage is limited. Canceling ends trial access immediately. You can
        manage your plan in billing.
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
