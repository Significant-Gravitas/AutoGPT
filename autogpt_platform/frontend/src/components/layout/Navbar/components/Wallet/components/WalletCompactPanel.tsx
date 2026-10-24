"use client";

import { useGetV1GetAutoTopUp } from "@/app/api/__generated__/endpoints/credits/credits";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { CreditCardIcon } from "@hugeicons/core-free-icons";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { useId } from "react";
import { TaskGroup } from "../helpers";
import { WalletCopilotUsage } from "./WalletCopilotUsage";
import { WalletEarnCredits } from "./WalletEarnCredits";
import { WalletTrialNote } from "./WalletTrialNote";

interface Props {
  groups: TaskGroup[];
  completedSteps: string[] | undefined;
  formattedCredits: string;
  trialNoteID: string;
  onAddCredits: () => void;
}

export function WalletCompactPanel({
  groups,
  completedSteps,
  formattedCredits,
  trialNoteID,
  onAddCredits,
}: Props) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const creditsHeadingID = useId();
  const { data: trial } = useTrialStatus();
  const { data: autoRefill } = useGetV1GetAutoTopUp({
    query: { select: okData },
  });

  return (
    <div className="flex flex-col">
      <WalletTrialNote id={trialNoteID} trial={trial} autoRefill={autoRefill} />
      <div className="px-5 py-4">
        <WalletCopilotUsage />
      </div>
      <section
        aria-labelledby={creditsHeadingID}
        className="mx-5 flex flex-col gap-4 border-t border-separator py-4"
      >
        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between gap-3">
            <Text
              as="h2"
              id={creditsHeadingID}
              variant="large-medium"
              tone="primary"
            >
              Automation credits
            </Text>
            <Text
              variant="large-semibold"
              tone="primary"
              className="tabular-nums"
            >
              {formattedCredits}
            </Text>
          </div>
          <Text variant="small" tone="secondary">
            Pays for workflow runs and paid steps, like making images.
          </Text>
        </div>
        {isPaymentEnabled && (
          <Button
            type="button"
            variant="secondary"
            size="small"
            leadingIcon={CreditCardIcon}
            onClick={onAddCredits}
            className="w-full"
          >
            Add credits
          </Button>
        )}
      </section>
      <WalletEarnCredits groups={groups} completedSteps={completedSteps} />
    </div>
  );
}
