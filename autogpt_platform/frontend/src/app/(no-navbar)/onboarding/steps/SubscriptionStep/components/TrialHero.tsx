import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatPlanPrice,
  trialPlanLabels,
} from "@/components/organisms/TrialCard/helpers";

interface Props {
  offer: TrialOfferResponse;
  onStart: () => void;
  isStarting: boolean;
  isDisabled: boolean;
  error: string | null;
}

export function TrialHero({
  offer,
  onStart,
  isStarting,
  isDisabled,
  error,
}: Props) {
  const days = offer.duration_days;
  return (
    <div className="flex flex-col items-center">
      <AutoGPTLogo hideText className="relative right-5 mb-2 h-8 w-20" />
      <Text variant="h2" as="h1">
        Start your {days}-day free trial
      </Text>
      <Text variant="large" tone="muted" className="mt-2 max-w-xl">
        Try {trialPlanLabels[offer.tier]} free for {days} days. Card required,
        nothing charged today. Then {formatPlanPrice(offer)} plus tax, cancel
        anytime.
      </Text>
      <Button
        type="button"
        size="large"
        className="mt-6 w-full max-w-xs"
        loading={isStarting}
        disabled={isStarting || isDisabled}
        onClick={onStart}
      >
        Start {days}-day free trial
      </Button>
      {error && (
        <Text variant="small" tone="error" role="alert" className="mt-2">
          {error}
        </Text>
      )}
      <Text variant="small" tone="muted" className="mt-10">
        Or compare plans
      </Text>
    </div>
  );
}
