import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { CheckoutAction } from "./CheckoutAction";
import { getTrialMilestones } from "./helpers";
import { TrialTimeline } from "./TrialTimeline";

// Once the trial status is in, its headline is the step's title.
export const TRIAL_TITLE_ID = "plan-step-title";

interface Props {
  offer: TrialOfferResponse;
  isStarting: boolean;
  isBlocked: boolean;
  error: string | null;
  onStart: () => void;
}

export function TrialPanel({
  offer,
  isStarting,
  isBlocked,
  error,
  onStart,
}: Props) {
  return (
    <TrialPanelShell>
      <header className="flex flex-col items-start gap-2">
        <Text variant="eyebrow" as="p">
          {offer.duration_days}-day free trial
        </Text>
        <Text variant="h3" as="h1" id={TRIAL_TITLE_ID} className="text-balance">
          Your first {offer.duration_days} days are on us
        </Text>
        <Text variant="large" tone="secondary" className="text-pretty">
          Nothing to pay today, and a reminder before there&apos;s ever anything
          to pay.
        </Text>
      </header>
      <TrialTimeline
        label="How your trial works"
        milestones={getTrialMilestones(offer)}
      />
      <CheckoutAction
        label="Start my free trial"
        lockLine="Stripe will ask for a card. Nothing is charged today."
        isProcessing={isStarting}
        isBlocked={isBlocked}
        error={error}
        onClick={onStart}
      />
    </TrialPanelShell>
  );
}

export function TrialPanelShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col gap-5 bg-surface px-5 pt-6 sm:px-8 lg:px-9 lg:py-8 [@media(max-height:780px)]:gap-4 lg:[@media(max-height:780px)]:py-6">
      {children}
    </div>
  );
}
