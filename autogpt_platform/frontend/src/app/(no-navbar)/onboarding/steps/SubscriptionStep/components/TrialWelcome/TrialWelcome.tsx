import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SquareLock02Icon } from "@hugeicons/core-free-icons";
import { ReadyLine } from "../ReadyLine";
import { Timeline } from "../Timeline";
import type { ReadyExpert } from "../useReadyExperts";
import { WelcomeShell } from "../WelcomeShell";
import { getTrialMilestones } from "./helpers";

interface Props {
  offer: TrialOfferResponse;
  experts: ReadyExpert[];
  isStarting: boolean;
  isBlocked: boolean;
  error: string | null;
  onStart: () => void;
}

export function TrialWelcome({
  offer,
  experts,
  isStarting,
  isBlocked,
  error,
  onStart,
}: Props) {
  return (
    <WelcomeShell art="letter">
      <header className="flex flex-col items-start gap-3">
        <ReadyLine experts={experts} />
        <Text variant="h3" as="h1" id="welcome-title" className="text-balance">
          Your first {offer.duration_days} days are on us
        </Text>
        <Text variant="large" tone="secondary" className="text-pretty">
          Nothing to pay today, and a reminder before there&apos;s ever anything
          to pay.
        </Text>
      </header>

      <Timeline
        label="How your trial works"
        milestones={getTrialMilestones(offer)}
      />

      <div className="mt-auto flex flex-col gap-3">
        <Button
          variant="primary"
          size="large"
          onClick={onStart}
          loading={isStarting}
          disabled={isStarting || isBlocked}
          className="h-12 w-full text-base"
        >
          Start my free trial
        </Button>
        {error ? (
          <Text variant="body" role="alert" tone="danger">
            {error}
          </Text>
        ) : null}
        <Text variant="small" tone="muted" className="text-balance text-center">
          <Icon
            icon={SquareLock02Icon}
            size={13}
            className="mr-1 inline-block -translate-y-px align-middle"
          />
          Stripe will ask for a card. Nothing is charged today.
        </Text>
      </div>
    </WelcomeShell>
  );
}
