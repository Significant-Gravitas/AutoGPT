"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "../../../helpers";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

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
}: Props) {
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
            {title ?? "Make room for what’s next."}
          </Text>
        </div>
      }
      styling={{ maxWidth: "520px" }}
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
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
        <div className="flex flex-col gap-4">
          <Text variant="body" as="span" className="text-zinc-700">
            {body}
          </Text>
        </div>

        {usageCarriesOver && (
          <Text
            variant="small"
            tone="secondary"
            className="mt-4 rounded-xl bg-zinc-50 p-4"
          >
            A higher allowance, with your current usage carried over. Your
            conversations and agents stay saved.
          </Text>
        )}
        <Dialog.Footer>
          <Button
            type="button"
            variant="ghost"
            size="large"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="primary"
            size="large"
            onClick={onConfirm}
            disabled={isSaving}
            loading={isSaving}
            data-fast-goal="subscription_change_confirm"
            data-fast-goal-surface="settings_billing"
          >
            {confirmLabel ?? `Upgrade to ${targetTierLabel}`}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
