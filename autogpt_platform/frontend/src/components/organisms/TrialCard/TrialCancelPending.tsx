import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatTrialEnd,
  formatTrialEndDate,
  formatTrialPrice,
  trialPlanLabels,
} from "./helpers";
import { TrialTitle } from "./TrialTitle/TrialTitle";

interface Props {
  trial: TrialStatusResponse;
  offer: TrialOfferResponse;
  isResuming: boolean;
  onResume: () => void;
}

export function TrialCancelPending({
  trial,
  offer,
  isResuming,
  onResume,
}: Props) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <TrialTitle>Your trial</TrialTitle>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-700">
          <span aria-hidden className="size-1.5 rounded-full bg-zinc-400" />
          Cancellation pending
        </span>
      </div>
      <Text variant="body" unmask={false} className="!text-zinc-800">
        Cancellation confirmed. Your trial will not convert to a paid plan and
        your card won&apos;t be charged. Trial access ends{" "}
        <strong className="font-semibold">
          {formatTrialEnd(trial.ends_at)}
        </strong>
        .
      </Text>
      {trial.ends_at ? (
        <Text variant="small" unmask={false} className="!text-zinc-500">
          Resume to keep the trial and start {trialPlanLabels[offer.tier]} on{" "}
          {formatTrialEndDate(trial.ends_at)} at {formatTrialPrice(offer)}, plus
          applicable tax.
        </Text>
      ) : null}
      <div className="mt-1">
        <Button
          variant="secondary"
          size="small"
          loading={isResuming}
          disabled={isResuming}
          onClick={onResume}
        >
          Resume trial
        </Button>
      </div>
    </div>
  );
}
