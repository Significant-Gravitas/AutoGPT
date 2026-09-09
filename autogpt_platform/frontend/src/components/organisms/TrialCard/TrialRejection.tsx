import type { TrialRejectionReason } from "@/app/api/__generated__/models/trialRejectionReason";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";

interface Props {
  reason: TrialRejectionReason;
}

export function TrialRejection({ reason }: Props) {
  const alreadyUsed = reason === "intro_offer_already_used";
  return (
    <div role="status" className="flex flex-col gap-3">
      <Text variant="h4">
        {alreadyUsed
          ? "This introductory offer has already been used"
          : "We couldn’t verify your card for this trial"}
      </Text>
      <Text variant="body">
        {alreadyUsed
          ? "This card or account has already redeemed an introductory offer. Each card and account can use one introductory offer."
          : "Your card could not be verified for trial eligibility, so this trial was not activated."}
      </Text>
      <Text variant="body">
        This trial will not convert to a paid subscription. You can choose a
        paid plan below. If you think this is a mistake, contact support.
      </Text>
      <Button
        as="NextLink"
        href="https://discord.gg/autogpt"
        target="_blank"
        rel="noopener noreferrer"
        variant="outline"
        size="small"
        className="self-start"
      >
        Contact support
      </Button>
    </div>
  );
}
