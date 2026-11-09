import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { CheckoutAction } from "./CheckoutAction";
import { getPaidMilestones } from "./helpers";
import { TRIAL_TITLE_ID, TrialPanelShell } from "./TrialPanel";
import { TrialTimeline } from "./TrialTimeline";

interface Props {
  price: string;
  cadence: "month" | "year";
  isProcessing: boolean;
  isBlocked: boolean;
  onContinue: () => void;
  notice?: ReactNode;
}

// No trial to offer: the plan beside a straight Pro checkout.
export function PaidPanel({
  price,
  cadence,
  isProcessing,
  isBlocked,
  onContinue,
  notice,
}: Props) {
  return (
    <TrialPanelShell>
      <header className="flex flex-col items-start gap-2">
        <Text variant="eyebrow" as="p">
          Pro plan
        </Text>
        <Text variant="h3" as="h1" id={TRIAL_TITLE_ID} className="text-balance">
          Start with Pro
        </Text>
        <Text variant="large" tone="secondary" className="text-pretty">
          {price} a {cadence}, plus any applicable tax. No contract, and you can
          cancel any time in Settings.
        </Text>
      </header>
      {notice}
      <TrialTimeline
        label="How your plan works"
        milestones={getPaidMilestones({ price, cadence })}
      />
      <CheckoutAction
        label="Continue to checkout"
        lockLine="Secure checkout by Stripe"
        isProcessing={isProcessing}
        isBlocked={isBlocked}
        onClick={onContinue}
      />
    </TrialPanelShell>
  );
}
