"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { useActiveOrgWallet } from "@/services/org-team/useActiveOrgWallet";
import { toast as sonnerToast } from "sonner";
import { TEAM_ORG_PLAN_REQUIRED } from "./paywall-detail";

export const PAYWALL_TOAST_KEY = "copilot-send-paywall";
const BILLING_PATH = "/settings/billing";

interface DescriptionProps {
  recovery: string;
  /** The backend's reason for the refusal, when it gave one. */
  detail?: string;
}

// Rendered inside the Toaster, so it can read whether this install bills at
// all and where the person works. Without billing (most self-hosted
// installs) there is no plan to pick and /settings/billing is a 404, so it
// points at whoever runs the server. In a team organization the
// organization's plan decides, so it says what the backend said instead of
// offering a personal plan.
export function PaywallToastDescription({
  recovery,
  detail,
}: DescriptionProps) {
  const isBillingEnabled = useIsBillingEnabled();
  const { isTeamOrg } = useActiveOrgWallet();
  if (isTeamOrg) {
    return `${detail ?? TEAM_ORG_PLAN_REQUIRED} ${asSentence(recovery)}`;
  }
  return isBillingEnabled
    ? `Choose one to keep chatting — ${recovery}.`
    : `Ask whoever runs this server to set one up — ${recovery}.`;
}

interface PlanRequiredProps {
  /** The backend's reason for the refusal, when it gave one. */
  detail?: string;
}

export function PlanRequiredDescription({ detail }: PlanRequiredProps) {
  const isBillingEnabled = useIsBillingEnabled();
  const { isTeamOrg } = useActiveOrgWallet();
  if (isTeamOrg) return detail ?? TEAM_ORG_PLAN_REQUIRED;
  return isBillingEnabled
    ? "Choose one to continue."
    : "Ask whoever runs this server to set one up.";
}

export function PaywallToastAction() {
  const isBillingEnabled = useIsBillingEnabled();
  const { isTeamOrg } = useActiveOrgWallet();
  if (!isBillingEnabled || isTeamOrg) return null;
  return (
    <Button
      as="NextLink"
      href={BILLING_PATH}
      size="small"
      variant="secondary"
      onClick={() => sonnerToast.dismiss(PAYWALL_TOAST_KEY)}
    >
      See plans
    </Button>
  );
}

function asSentence(clause: string) {
  return `${clause.charAt(0).toUpperCase()}${clause.slice(1)}.`;
}
