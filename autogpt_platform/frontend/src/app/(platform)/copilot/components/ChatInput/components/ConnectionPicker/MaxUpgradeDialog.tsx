"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { MaxUpgradeBenefits } from "./MaxUpgradeBenefits";
import { MaxUpgradePricing } from "./MaxUpgradePricing";
import { formatUpgradePrice } from "./maxUpgrade";
import type { useMaxUpgrade } from "./useMaxUpgrade";

type Upgrade = ReturnType<typeof useMaxUpgrade>;
type Props = {
  isOpen: boolean;
  onClose: () => void;
  onCloseAutoFocus: (event: Event) => void;
  model?: string | null;
  upgrade: Upgrade;
};

export function MaxUpgradeDialog({
  isOpen,
  onClose,
  onCloseAutoFocus,
  model,
  upgrade,
}: Props) {
  const [reviewing, setReviewing] = useState(false);
  function close() {
    if (upgrade.isPending) return;
    setReviewing(false);
    upgrade.resetError();
    onClose();
  }
  async function confirm() {
    const success = await upgrade.upgrade();
    setReviewing(false);
    if (success) onClose();
  }
  return (
    <Dialog
      title={
        <span className="font-sans text-2xl font-medium leading-tight tracking-tight">
          {reviewing ? "Upgrade to Max?" : "Unlock Advanced with Max."}
        </span>
      }
      className="min-w-0 rounded-2xl text-zinc-900 lg:max-w-[29rem]"
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) close();
        },
      }}
      forceOpen={isOpen && upgrade.isPending}
      onCloseAutoFocus={onCloseAutoFocus}
    >
      <Dialog.Content>
        {reviewing ? (
          <MaxUpgradeConfirmation
            upgrade={upgrade}
            onCancel={() => setReviewing(false)}
            onConfirm={confirm}
          />
        ) : (
          <>
            <MaxUpgradeBenefits model={model} />
            <MaxUpgradePricing pricing={upgrade.pricing} />
            {(upgrade.isError || upgrade.error) && (
              <div role="alert" className="mt-4">
                <ErrorCard
                  context="subscription"
                  responseError={{
                    message:
                      upgrade.error || "We couldn’t load your plan details.",
                  }}
                  onRetry={upgrade.isError ? upgrade.retry : undefined}
                />
              </div>
            )}
            {(upgrade.unavailableReason || upgrade.error) && (
              <div className="mt-4 space-y-2 text-xs text-zinc-600">
                {upgrade.unavailableReason && (
                  <p>{upgrade.unavailableReason}</p>
                )}
                <BillingLink />
              </div>
            )}
            <div className="mt-5 border-t border-zinc-200 pt-5">
              <Button
                type="button"
                autoFocus
                size="small"
                className="h-11 w-full rounded-lg border-purple-600 bg-purple-600 hover:border-purple-700 hover:bg-purple-700"
                disabled={
                  !upgrade.canUpgrade || upgrade.isLoading || upgrade.isError
                }
                onClick={() => {
                  upgrade.resetError();
                  setReviewing(true);
                }}
              >
                Review upgrade
                <Icon icon={ArrowRight02Icon} size={16} aria-hidden />
              </Button>
              <p className="mt-2.5 text-center text-[11px] text-zinc-600">
                You’ll confirm the change before upgrading.
              </p>
              <Button
                type="button"
                variant="ghost"
                size="small"
                className="mt-1 w-full text-xs text-zinc-600"
                onClick={close}
              >
                Keep using Pro
              </Button>
            </div>
          </>
        )}
      </Dialog.Content>
    </Dialog>
  );
}

function MaxUpgradeConfirmation({
  upgrade,
  onCancel,
  onConfirm,
}: {
  upgrade: Upgrade;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const pricing = upgrade.pricing;
  const period = pricing?.cycle === "yearly" ? "year" : "month";
  return (
    <div className="space-y-4 text-sm leading-relaxed text-zinc-600">
      <p>
        Your saved payment method will be charged the prorated difference
        immediately. Your existing billing cycle stays the same.
      </p>
      {pricing?.maxCents && (
        <p>
          The standard Max price is{" "}
          <span className="font-medium text-zinc-900">
            {formatUpgradePrice(pricing.maxCents)} per {period}
          </span>
          , before taxes and any applicable discounts.
        </p>
      )}
      {!upgrade.canUpgrade && <p role="alert">{upgrade.unavailableReason}</p>}
      <Dialog.Footer>
        <Button
          type="button"
          autoFocus
          variant="ghost"
          size="small"
          onClick={onCancel}
          disabled={upgrade.isPending}
        >
          Cancel
        </Button>
        <Button
          type="button"
          size="small"
          className="rounded-lg border-purple-600 bg-purple-600 hover:border-purple-700 hover:bg-purple-700"
          disabled={!upgrade.canUpgrade || upgrade.isPending}
          loading={upgrade.isPending}
          onClick={onConfirm}
          data-fast-goal="subscription_change_confirm"
          data-fast-goal-surface="model_picker"
        >
          Upgrade to Max
        </Button>
      </Dialog.Footer>
    </div>
  );
}

function BillingLink() {
  return (
    <Link
      href="/settings/billing"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex underline underline-offset-4"
    >
      Open billing in a new tab
    </Link>
  );
}
