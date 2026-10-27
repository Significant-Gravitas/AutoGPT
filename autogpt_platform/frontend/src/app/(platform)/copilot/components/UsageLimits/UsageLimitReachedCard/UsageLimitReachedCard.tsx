"use client";

import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { getUsageOfferAction } from "@/services/usageExperience/helpers";
import { formatPaymentDue } from "@/services/usageExperience/payment";
import { usagePresentation } from "@/services/usageExperience/presentation";
import { PaymentDueAction } from "@/components/organisms/UsageExperience/PaymentDueAction";
import { ProviderContinuation } from "@/components/organisms/UsageExperience/ProviderContinuation";
import { UsageMeters } from "@/components/organisms/UsageExperience/UsageMeters";
import {
  budgetOutTitle,
  budgetResetSentence,
} from "@/services/usageExperience/budget";
import type { UsageNoticeContext } from "./useUsageLimitReachedCard";
import { useUsageLimitReachedCard } from "./useUsageLimitReachedCard";

export function UsageLimitReachedCard(props: UsageNoticeContext = {}) {
  const state = useUsageLimitReachedCard(props);
  // Relative to the clock, so read on every render rather than cached.
  const resetSentence = useReadEveryRender(() =>
    state.wait ? budgetResetSentence(state.wait) : null,
  );
  if (state.isLoading || (!state.experience.blocked && !state.isError))
    return null;
  const copy = usagePresentation(state.experience, state.offer.usageMultiplier);
  const isOffer =
    state.isBillingEnabled && state.experience.targetTier && !state.isError;
  const due = formatPaymentDue(state.experience.payment);
  const payment = state.isBillingEnabled && !state.isError ? due : null;
  return (
    <section
      role="status"
      aria-label="Usage notice"
      className="mx-auto w-full rounded-2xl border border-border-base bg-surface p-5 shadow-sm"
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <Text variant="small" className="!text-xs !text-foreground-tertiary">
          {state.isRefreshing
            ? "Checking your allowance"
            : state.isError
              ? "Usage temporarily unavailable"
              : copy.eyebrow}
        </Text>
        {!state.isError && (
          <Button
            as="NextLink"
            href="/settings/billing"
            size="small"
            variant="ghost"
            className="!h-auto !p-0 !text-xs !text-foreground-tertiary"
          >
            {state.experience.inactiveTrialStatus === "payment_failed"
              ? "Review billing"
              : "View usage"}
          </Button>
        )}
      </div>
      <Text variant="h4" className="!text-lg !leading-6">
        {state.isRefreshing
          ? "Checking your usage"
          : state.isError
            ? "We couldn’t check your usage."
            : state.wait
              ? budgetOutTitle(state.wait.window)
              : copy.noticeTitle}
      </Text>
      <Text
        variant="small"
        className="mt-2 !leading-5 !text-foreground-tertiary"
      >
        {state.isRefreshing
          ? "Refreshing your plan and usage…"
          : state.isError
            ? "Your work is saved. Try again to check your allowance."
            : state.wait
              ? resetSentence
              : copy.description}
      </Text>
      {state.wait && (
        <UsageMeters
          className="mt-4"
          daily={state.usage?.daily}
          weekly={state.usage?.weekly}
        />
      )}
      {isOffer && (
        <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2">
          <Button
            size="small"
            onClick={state.upgrade}
            disabled={state.offer.disabled}
            rightIcon={<Icon icon={ArrowRight02Icon} size={14} />}
          >
            {getUsageOfferAction(state.experience, state.offer)}
          </Button>
          {state.experience.freshProUsage && (
            <Text
              variant="small"
              className="!text-xs !text-foreground-tertiary"
            >
              + tax. Review before you pay.
            </Text>
          )}
        </div>
      )}
      {payment && (
        <div className="mt-4">
          <PaymentDueAction
            url={payment.url}
            label={payment.actionLabel}
            size="small"
            layout="inline"
          />
        </div>
      )}
      {state.alternative && (
        <ProviderContinuation
          name={state.alternative.display_name}
          onContinue={state.continueHere}
          isSwitching={state.isSwitching}
        />
      )}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
        <Text variant="small" className="!text-xs !text-foreground-subtle">
          You can keep editing your draft.
        </Text>
        {state.isError ? (
          <Button
            size="small"
            variant="ghost"
            onClick={() => void state.retry()}
            disabled={state.isRefreshing}
          >
            {state.isRefreshing ? "Checking…" : "Try again"}
          </Button>
        ) : state.experience.isTopTier && state.isBillingEnabled ? (
          <Button
            as="NextLink"
            href="mailto:contact@agpt.co"
            size="small"
            variant="ghost"
          >
            {copy.supportLabel}
          </Button>
        ) : null}
      </div>
    </section>
  );
}
