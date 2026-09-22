import type { TrialRejectionReason } from "@/app/api/__generated__/models/trialRejectionReason";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { TrialTitle } from "./TrialTitle/TrialTitle";
import { trialRejectionCopy } from "./helpers";

interface Props {
  reason: TrialRejectionReason;
}

export function TrialRejection({ reason }: Props) {
  const { title, detail } = trialRejectionCopy(reason);
  return (
    <div role="status" className="flex flex-col gap-3">
      <TrialTitle>{title}</TrialTitle>
      <Text variant="body" className="!text-zinc-800">
        {detail}
      </Text>
      <Text variant="body" className="!text-zinc-800">
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
