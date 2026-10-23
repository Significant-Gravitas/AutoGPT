"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "../../../helpers";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { formatUpgradeAmount } from "./MaxUpgradeSummary";
import { MaxUpgradeReview } from "./MaxUpgradeReview";
import type { useTierUpgradePreview } from "./useTierUpgradePreview";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  targetTierLabel: string;
  body: string;
  isSaving: boolean;
  onConfirm: () => void;
  title?: string;
  confirmLabel?: string;
  priceCents?: number;
  billingCycle?: "monthly" | "yearly";
  usageCarriesOver?: boolean;
  usageMultiplier?: number;
  upgradeReview?: ReturnType<typeof useTierUpgradePreview>;
}

export function SwitchTierDialog({
  isOpen,
  onOpenChange,
  targetTierLabel,
  body,
  isSaving,
  onConfirm,
  title,
  confirmLabel,
  priceCents,
  billingCycle = "monthly",
  usageCarriesOver = false,
  usageMultiplier,
  upgradeReview,
}: Props) {
  const preview = upgradeReview?.preview;
  const reviewing = upgradeReview?.required;
  const cannotConfirm = isSaving || (reviewing && !preview);
  const action = preview
    ? `Pay ${formatUpgradeAmount(preview.amount_due, preview.currency)} and upgrade`
    : (confirmLabel ?? `Upgrade to ${targetTierLabel}`);
  return (
    <Dialog
      variant="compact"
      title={
        <div className="space-y-2 pr-7">
          <Text variant="small" className="text-zinc-500">
            {title ? "Your subscription" : `Upgrade to ${targetTierLabel}`}
          </Text>
          <Text
            variant="h3"
            as="span"
            className="block text-[26px] leading-8 tracking-tight"
          >
            {title ?? `Upgrade to ${targetTierLabel}`}
          </Text>
        </div>
      }
      styling={{ maxWidth: "520px" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        {usageCarriesOver && (
          <Text variant="body" className="mb-5 leading-6 text-zinc-600">
            {usageMultiplier
              ? `${usageMultiplier} times the usage of Pro and priority support.`
              : "More room for demanding work and priority support."}
          </Text>
        )}
        {reviewing ? (
          <MaxUpgradeReview review={upgradeReview} />
        ) : (
          <>
            {priceCents !== undefined && (
              <div className="mb-5 flex flex-wrap items-baseline justify-between gap-3 rounded-2xl border border-zinc-200 bg-zinc-50 p-5">
                <Text variant="large-medium">AutoGPT {targetTierLabel}</Text>
                <div>
                  <Text variant="h3" as="span">
                    {formatCents(priceCents)}
                  </Text>
                  <Text variant="small" as="span" tone="secondary">
                    {" "}
                    / {billingCycle === "yearly" ? "year" : "month"}
                  </Text>
                </div>
              </div>
            )}
            <Text variant="body" as="span" className="text-zinc-700">
              {body}
            </Text>
          </>
        )}
        {usageCarriesOver && (
          <Text
            variant="small"
            tone="secondary"
            className="mt-4 rounded-xl bg-zinc-50 p-4 leading-6"
          >
            You can keep going once payment goes through. What you’ve already
            used counts toward Max’s limit
            {usageMultiplier
              ? `, which is ${usageMultiplier} times as high as Pro’s.`
              : "."}
          </Text>
        )}
        <Dialog.Footer
          className={
            usageCarriesOver ? "flex-col items-stretch gap-2" : undefined
          }
        >
          {usageCarriesOver ? (
            <>
              <ConfirmButton
                label={action}
                disabled={cannotConfirm}
                isSaving={isSaving}
                onConfirm={onConfirm}
              />
              <Button
                type="button"
                variant="ghost"
                size="large"
                className="w-full"
                onClick={() => onOpenChange(false)}
                disabled={isSaving}
              >
                Keep Pro
              </Button>
            </>
          ) : (
            <>
              <Button
                type="button"
                variant="ghost"
                size="large"
                onClick={() => onOpenChange(false)}
                disabled={isSaving}
              >
                Cancel
              </Button>
              <ConfirmButton
                label={action}
                disabled={cannotConfirm}
                isSaving={isSaving}
                onConfirm={onConfirm}
              />
            </>
          )}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

function ConfirmButton({
  label,
  disabled,
  isSaving,
  onConfirm,
}: {
  label: string;
  disabled?: boolean;
  isSaving: boolean;
  onConfirm: () => void;
}) {
  return (
    <Button
      type="button"
      variant="primary"
      size="large"
      className="w-full"
      onClick={onConfirm}
      disabled={disabled}
      loading={isSaving}
    >
      {label}
    </Button>
  );
}
