"use client";

import { ArrowRight02Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { usagePresentation } from "@/services/usageExperience/presentation";
import { ProviderContinuation } from "@/components/organisms/UsageExperience/ProviderContinuation";
import type { UsageNoticeContext } from "./useUsageLimitReachedCard";
import { useUsageLimitReachedCard } from "./useUsageLimitReachedCard";

export function UsageLimitReachedCard(props: UsageNoticeContext = {}) {
  const state = useUsageLimitReachedCard(props);
  if (state.isLoading || (!state.experience.blocked && !state.isError))
    return null;
  const copy = usagePresentation(state.experience);
  const isOffer =
    state.isBillingEnabled && state.experience.targetTier && !state.isError;
  return (
    <section
      role="status"
      aria-label="Usage notice"
      className="mx-auto w-full overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm"
    >
      <div className="px-5 py-4">
        <Text variant="small" className="mb-2 !text-xs !text-zinc-500">
          {state.isRefreshing
            ? "Checking your allowance"
            : state.isError
              ? "Usage temporarily unavailable"
              : copy.eyebrow}
        </Text>
        <Text variant="h4" className="!text-lg !leading-6">
          {state.isRefreshing
            ? "One moment. Your work is saved."
            : state.isError
              ? "We couldn’t check your usage."
              : copy.noticeTitle}
        </Text>
        <Text variant="small" className="mt-2 !leading-5 !text-zinc-500">
          {state.isRefreshing
            ? "Refreshing your plan and usage…"
            : state.isError
              ? "Your work is saved. Try again to check your allowance."
              : state.experience.inactiveTrialStatus
                ? copy.description
                : copy.trialSpent
                  ? state.experience.freshProUsage
                    ? "Continue with Pro. Your work stays right here, with fresh usage when your plan activates."
                    : "Your trial allowance is used. Choose a plan to continue with your work."
                  : copy.resetLabel
                    ? `Your allowance refreshes ${copy.resetLabel}. Your work is saved.`
                    : "Your chats, agents, and results are still here."}
        </Text>
      </div>
      {isOffer && (
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-purple-100 bg-gradient-to-r from-purple-50 to-blue-50 px-5 py-4">
          <div className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-xl border border-purple-200 bg-white text-purple-600">
              <Icon icon={SparklesIcon} size={18} />
            </span>
            <div>
              <Text variant="small-medium">
                {state.experience.targetTier === "PRO"
                  ? "Keep going with Pro"
                  : "More room with Max"}
              </Text>
              <Text variant="small" className="mt-0.5 !text-xs !text-zinc-500">
                {state.experience.freshProUsage
                  ? "Fresh allowance. Same conversation."
                  : state.experience.inactiveTrialStatus
                    ? "Your work is saved whenever you’re ready."
                    : "Your existing usage carries over."}
              </Text>
            </div>
          </div>
          <Button
            size="small"
            onClick={state.upgrade}
            disabled={state.offer.disabled}
            rightIcon={<Icon icon={ArrowRight02Icon} size={14} />}
          >
            {state.experience.targetTier === "PRO"
              ? "Upgrade to Pro"
              : "Review Max upgrade"}
          </Button>
        </div>
      )}
      {state.alternative && (
        <div className="px-5 pb-3">
          <ProviderContinuation
            name={state.alternative.display_name}
            onContinue={state.continueHere}
            isSwitching={state.isSwitching}
          />
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-zinc-100 px-5 py-3">
        <Text variant="small" className="!text-xs !text-zinc-400">
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
        ) : (
          <Button
            as="NextLink"
            href="/settings/billing"
            size="small"
            variant="ghost"
          >
            {state.experience.inactiveTrialStatus === "payment_failed"
              ? "Review billing"
              : "View usage"}
          </Button>
        )}
      </div>
    </section>
  );
}
