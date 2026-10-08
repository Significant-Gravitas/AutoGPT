"use client";

import type { SubscriptionTier } from "@/app/api/__generated__/models/subscriptionTier";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { PlanOffer } from "@/components/organisms/UsageExperience/PlanOffer";
import { ProviderContinuation } from "@/components/organisms/UsageExperience/ProviderContinuation";
import { UsageMeter } from "@/components/organisms/UsageExperience/UsageMeter";
import {
  getUsageExperience,
  type UsageExperience,
} from "@/services/usageExperience/helpers";
import { usagePresentation } from "@/services/usageExperience/presentation";
import { useRouter } from "next/navigation";
import type { ComponentProps } from "react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  resetsAt?: string | Date | null;
  tier?: SubscriptionTier | null;
  experience?: UsageExperience;
  offer?: Omit<ComponentProps<typeof PlanOffer>, "onUpgrade">;
  onUpgrade?: () => void;
  unavailable?: boolean;
  checking?: boolean;
  failureWindow?: string;
  onRetry?: () => void;
  isBillingEnabled?: boolean;
  alternative?: { display_name: string } | null;
  onContinue?: () => void;
  isSwitching?: boolean;
}

export function RateLimitResetDialog({
  isOpen,
  onClose,
  resetsAt,
  tier,
  experience,
  offer,
  onUpgrade,
  unavailable = false,
  checking = false,
  failureWindow,
  onRetry,
  isBillingEnabled = true,
  alternative,
  onContinue,
  isSwitching,
}: Props) {
  const router = useRouter();
  const model =
    experience ??
    getUsageExperience({
      tier: tier ?? "BASIC",
      daily: {
        percent_used: 100,
        resets_at: resetsAt ? new Date(resetsAt).toISOString() : "",
      },
    });
  const copy = usagePresentation(model);
  const fallbackTitle =
    failureWindow === "trial"
      ? "Trial allowance reached."
      : failureWindow === "weekly"
        ? "Weekly usage reached."
        : failureWindow === "daily"
          ? "Daily usage reached."
          : "We couldn’t check your usage.";
  function upgrade() {
    if (onUpgrade) onUpgrade();
    else {
      onClose();
      router.push("/settings/billing");
    }
  }
  return (
    <Dialog
      variant="compact"
      title={
        <div className="pr-6">
          <Text variant="small" className="mb-2 !text-xs !text-zinc-500">
            {checking
              ? "Checking your allowance"
              : unavailable
                ? "Usage temporarily unavailable"
                : copy.eyebrow}
          </Text>
          <Text as="span" variant="h3" className="!text-[26px] !leading-8">
            {checking
              ? "One moment. Your work is saved."
              : unavailable
                ? fallbackTitle
                : copy.title}
          </Text>
        </div>
      }
      styling={{ maxWidth: "35rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        {checking ? (
          <Text
            variant="body"
            role="status"
            className="!text-sm !text-zinc-500"
          >
            Refreshing your plan and usage…
          </Text>
        ) : unavailable ? (
          <div className="space-y-5">
            <Text variant="body" className="!text-sm !leading-6 !text-zinc-500">
              Your work is saved. Try again to check your allowance before
              continuing.
            </Text>
            <Button onClick={onRetry}>Try again</Button>
          </div>
        ) : (
          <>
            <Text
              variant="body"
              className="mb-5 !text-sm !leading-6 !text-zinc-500"
            >
              {copy.description}
            </Text>
            {copy.trialSpent ? (
              <div className="mb-5">
                <UsageMeter
                  label="Trial allowance"
                  percent={model.trialPercent ?? 100}
                  detail="One allowance for your trial. It doesn’t refresh."
                />
              </div>
            ) : (
              copy.resetLabel && (
                <div className="mb-5 rounded-xl border border-zinc-200 bg-zinc-50 p-4">
                  <Text variant="small" className="mb-1 !text-zinc-500">
                    You can continue after
                  </Text>
                  <Text variant="body-medium">{copy.resetLabel}</Text>
                </div>
              )
            )}
            {isBillingEnabled && model.targetTier && (
              <PlanOffer
                tier={model.targetTier}
                {...offer}
                onUpgrade={upgrade}
              />
            )}
            {isBillingEnabled && model.isTopTier && (
              <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-zinc-200 p-4">
                <div>
                  <Text variant="small-medium">Need more capacity?</Text>
                  <Text variant="small" className="mt-1 !text-zinc-500">
                    We’ll help find the right fit.
                  </Text>
                </div>
                <Button
                  as="NextLink"
                  href="mailto:contact@agpt.co"
                  variant="secondary"
                  size="small"
                >
                  {copy.supportLabel}
                </Button>
              </div>
            )}
          </>
        )}
        {alternative && onContinue && (
          <ProviderContinuation
            name={alternative.display_name}
            onContinue={onContinue}
            isSwitching={isSwitching}
          />
        )}
        <Button variant="ghost" className="mt-3 w-full" onClick={onClose}>
          {checking || unavailable ? "Back to my work" : copy.secondary}
        </Button>
      </Dialog.Content>
    </Dialog>
  );
}
