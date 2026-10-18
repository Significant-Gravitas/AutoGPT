"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { toast as sonnerToast } from "sonner";

export const PAYWALL_TOAST_KEY = "copilot-send-paywall";
const BILLING_PATH = "/settings/billing";

interface DescriptionProps {
  recovery: string;
}

// Rendered inside the Toaster, so it can read whether this install bills at
// all. Without billing (most self-hosted installs) there is no plan to pick
// and /settings/billing is a 404, so it points at whoever runs the server.
export function PaywallToastDescription({ recovery }: DescriptionProps) {
  const isBillingEnabled = useIsBillingEnabled();
  return isBillingEnabled
    ? `Choose one to keep chatting — ${recovery}.`
    : `Ask whoever runs this server to set one up — ${recovery}.`;
}

export function PaywallToastAction() {
  const isBillingEnabled = useIsBillingEnabled();
  if (!isBillingEnabled) return null;
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
