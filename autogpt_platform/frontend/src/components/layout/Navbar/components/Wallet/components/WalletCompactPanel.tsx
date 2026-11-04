"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { usePlatformPayment } from "@/hooks/usePlatformPayment";
import {
  ArrowRight01Icon,
  CreditCardIcon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import {
  ORG_CREDITS_MANAGED_MESSAGE,
  TEAM_ORG_TOP_UP_REFUSED_MESSAGE,
} from "@/services/org-team/org-wallet";
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
  isCreditsForbidden?: boolean;
  isTeamOrg?: boolean;
  trialNoteID: string;
  onAddCredits: () => void;
  onOpenExplainer: () => void;
  onNavigate: () => void;
}

export function WalletCompactPanel({
  groups,
  completedSteps,
  formattedCredits,
  isCreditsForbidden = false,
  isTeamOrg = false,
  trialNoteID,
  onAddCredits,
  onOpenExplainer,
  onNavigate,
}: Props) {
  const { isPaymentEnabled } = usePlatformPayment();
  const creditsHeadingID = useId();
  const { data: trial } = useTrialStatus();

  return (
    <div className="flex flex-col">
      <WalletTrialNote id={trialNoteID} trial={trial} />
      <div className="px-5 py-4">
        <WalletCopilotUsage />
      </div>
      <section
        aria-labelledby={creditsHeadingID}
        className="mx-5 flex flex-col gap-4 border-t border-separator py-4"
      >
        <div className="flex items-baseline justify-between gap-3">
          <Text
            as="h2"
            id={creditsHeadingID}
            variant="large-medium"
            tone="primary"
          >
            Automation credits
          </Text>
          {isCreditsForbidden ? null : (
            <Text variant="large-semibold" tone="primary" tabular>
              {formattedCredits}
            </Text>
          )}
        </div>
        {isCreditsForbidden ? (
          <Text variant="small" tone="muted">
            {ORG_CREDITS_MANAGED_MESSAGE}
          </Text>
        ) : isPaymentEnabled && isTeamOrg ? (
          <Text variant="small" tone="muted">
            {TEAM_ORG_TOP_UP_REFUSED_MESSAGE}
          </Text>
        ) : isPaymentEnabled ? (
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
        ) : null}
      </section>
      <WalletEarnCredits groups={groups} completedSteps={completedSteps} />
      <div className="sticky bottom-0 flex items-center justify-between gap-2 border-t border-separator bg-surface px-3 py-2">
        <Button
          type="button"
          variant="ghost"
          size="xs"
          leadingIcon={InformationCircleIcon}
          onClick={onOpenExplainer}
        >
          How it works
        </Button>
        <Button
          as="NextLink"
          href="/settings/billing"
          variant="ghost"
          size="xs"
          rightIcon={<Icon icon={ArrowRight01Icon} size={14} aria-hidden />}
          onClick={onNavigate}
        >
          Billing settings
        </Button>
      </div>
    </div>
  );
}
