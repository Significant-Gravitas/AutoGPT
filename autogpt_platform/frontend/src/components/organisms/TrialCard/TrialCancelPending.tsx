import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatPlanPrice,
  formatTrialEnd,
  formatTrialEndDate,
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
    <div data-slot="trial-cancel-pending" className="flex flex-col gap-3">
      <div
        data-slot="trial-cancel-pending-header"
        className="flex flex-wrap items-center gap-2"
      >
        <TrialTitle>Your trial</TrialTitle>
        <span
          data-slot="trial-cancel-pending-status"
          className="inline-flex items-center gap-1.5 rounded-full bg-surface-hover px-2 py-0.5 text-xs text-foreground-body"
        >
          <span
            // TODO(tokens): no surface role for the zinc-400 status dot.
            aria-hidden
            data-slot="trial-cancel-pending-status-dot"
            className="size-1.5 rounded-full bg-zinc-400"
          />
          Cancellation pending
        </span>
      </div>
      <Text variant="body" className="!text-foreground-emphasis">
        Cancellation confirmed. Your trial will not convert to a paid plan and
        your card won&apos;t be charged. Trial access ends{" "}
        <strong
          data-slot="trial-cancel-pending-end-date"
          className="font-semibold"
        >
          {formatTrialEnd(trial.ends_at)}
        </strong>
        .
      </Text>
      {trial.ends_at ? (
        <Text variant="small" tone="muted">
          Resume to keep the trial and start {trialPlanLabels[offer.tier]} on{" "}
          {formatTrialEndDate(trial.ends_at)} at {formatPlanPrice(offer)}, plus
          applicable tax.
        </Text>
      ) : null}
      <div data-slot="trial-cancel-pending-actions" className="mt-1">
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
