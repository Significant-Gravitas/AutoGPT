"use client";

import type { SubscriptionTier } from "@/app/api/__generated__/models/subscriptionTier";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useRouter } from "next/navigation";
import { formatResetTime } from "../usageHelpers";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  resetsAt?: string | Date | null;
  tier?: SubscriptionTier | null;
  /**
   * A linked subscription this chat could continue on instead. The platform
   * cap does not apply to a turn billed to the user's own credential, so when
   * the account holds one, switching is a way out that costs nothing more.
   * Offered beside the upgrade, never instead of it: the cap is still ours.
   */
  alternative?: { display_name: string } | null;
  onContinue?: () => void;
  isSwitching?: boolean;
}

const CONTACT_US_URL = "mailto:contact@agpt.co";
const BILLING_PATH = "/settings/billing";

// Tiers that have no higher self-serve plan to upgrade to — direct these
// users to support instead of the billing page.
const TOP_TIERS: ReadonlySet<SubscriptionTier> = new Set([
  "MAX",
  "BUSINESS",
  "ENTERPRISE",
]);

export function RateLimitResetDialog({
  isOpen,
  onClose,
  resetsAt,
  tier,
  alternative = null,
  onContinue,
  isSwitching = false,
}: Props) {
  const router = useRouter();
  const resetTimeLabel = resetsAt ? formatResetTime(resetsAt) : null;
  const isTopTier = !!tier && TOP_TIERS.has(tier);
  const canContinue = alternative !== null && onContinue !== undefined;

  const ctaLabel = isTopTier ? "Contact us" : "Upgrade plan";
  const bodyTrailer = isTopTier
    ? "or contact us if you need more capacity."
    : "or upgrade your plan.";

  function handleCtaClick() {
    onClose();
    if (isTopTier) {
      window.open(CONTACT_US_URL, "_blank", "noopener,noreferrer");
      return;
    }
    router.push(BILLING_PATH);
  }

  return (
    <Dialog
      title="Daily usage limit reached"
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          You&apos;ve reached your daily usage limit.
          {resetTimeLabel && resetTimeLabel !== "now"
            ? ` Resets ${resetTimeLabel}.`
            : ""}{" "}
          You can still browse, edit agents, and view results &mdash;{" "}
          {bodyTrailer}
        </Text>
        {canContinue && (
          <Text variant="small" className="mt-3 !text-zinc-500">
            Or continue on {alternative.display_name}: the rest of this chat
            runs there instead, and everything already said stays as it is.
          </Text>
        )}
        <Dialog.Footer className="!justify-center">
          <Button variant="secondary" onClick={onClose}>
            Wait for reset
          </Button>
          <Button
            variant={canContinue ? "secondary" : "primary"}
            onClick={handleCtaClick}
          >
            {ctaLabel}
          </Button>
          {canContinue && (
            <Button
              variant="primary"
              onClick={onContinue}
              loading={isSwitching}
            >
              Continue on {alternative.display_name}
            </Button>
          )}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
