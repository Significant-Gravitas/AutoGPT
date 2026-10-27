import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatTrialPrice, trialPlanLabels } from "./helpers";
import { TrialTitle } from "./TrialTitle/TrialTitle";

interface Props {
  trial: TrialStatusResponse;
  isStarting: boolean;
  isBlocked: boolean;
  onStart: () => void;
}

export function TrialOffer({ trial, isStarting, isBlocked, onStart }: Props) {
  const offer = trial.offer;
  if (!offer) return null;
  return (
    <div
      data-slot="trial-offer"
      className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between md:gap-8"
    >
      <div
        data-slot="trial-offer-details"
        className="flex min-w-0 flex-col gap-2"
      >
        <div
          data-slot="trial-offer-header"
          className="flex flex-wrap items-center gap-2"
        >
          <TrialTitle>
            Try AutoGPT {trialPlanLabels[offer.tier]} for {offer.duration_days}{" "}
            days
          </TrialTitle>
          <span
            data-slot="trial-offer-pill"
            // TODO(tokens): no role for the purple-100 pill or its purple-700
            // text.
            className="inline-flex items-center rounded-full bg-purple-100 px-2 py-0.5 text-tiny font-medium leading-inherit text-purple-700"
          >
            No charge today
          </span>
        </div>
        <Text variant="body" className="!text-foreground-emphasis">
          Card required. No subscription charge today. Then{" "}
          {formatTrialPrice(offer)}, plus applicable tax, unless you cancel
          before the trial ends.
        </Text>
        <Text variant="small" tone="muted">
          Trial usage is limited. If you cancel, you keep access until the trial
          ends. You can manage your plan in billing.
        </Text>
      </div>
      <Button
        variant="primary"
        size="large"
        onClick={onStart}
        loading={isStarting}
        disabled={isStarting || isBlocked}
        className="w-full shrink-0 md:w-auto md:min-w-[11rem]"
      >
        Start {offer.duration_days}-day trial
      </Button>
    </div>
  );
}
