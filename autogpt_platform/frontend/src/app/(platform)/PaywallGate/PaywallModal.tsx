"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { PlanCard } from "@/components/molecules/PlanCard/PlanCard";
import { cn } from "@/lib/utils";
import { SwitchTierDialog } from "../settings/billing/components/SubscriptionTab/YourPlanCard/SwitchTierDialog";
import { usePaywallModal } from "./usePaywallModal";

// Non-dismissable Stripe paywall for NO_TIER users. Reuses the onboarding
// PlanCard + Monthly/Yearly toggle so both surfaces share one visual.
export function PaywallModal() {
  const {
    isLoading,
    plans,
    country,
    isYearly,
    selectedCycle,
    setSelectedCycle,
    handleSelectPlan,
    isPending,
    selectedTier,
    pendingTier,
    pendingTierLabel,
    confirmPendingTier,
    cancelPendingTier,
  } = usePaywallModal();

  return (
    <Dialog forceOpen controlled={{ isOpen: true, set: () => {} }}>
      <Dialog.Content>
        <div className="flex w-full flex-col items-center gap-4 px-2 py-2">
          <div className="flex flex-col items-center gap-1 text-center">
            <Text
              variant="h3"
              className="!text-[1.375rem] !leading-[1.6rem] md:!text-[1.75rem] md:!leading-[2.5rem]"
            >
              Choose the plan that&apos;s right for{" "}
              <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
                you
              </span>
            </Text>
            <Text variant="body" className="!text-zinc-500">
              Pick a plan to unlock AutoPilot and start running agents.
            </Text>
          </div>

          {plans.length > 0 && (
            <div
              role="radiogroup"
              aria-label="Billing cycle"
              className="inline-flex rounded-full border border-[#d8d8d8] bg-zinc-100 p-[3px]"
            >
              {(["monthly", "yearly"] as const).map((cycle) => (
                <button
                  key={cycle}
                  role="radio"
                  aria-checked={selectedCycle === cycle}
                  type="button"
                  onClick={() => setSelectedCycle(cycle)}
                  className={cn(
                    "rounded-full border-none px-4 py-1.5 text-xs font-medium transition-all",
                    selectedCycle === cycle
                      ? "bg-white text-zinc-900 shadow-sm"
                      : "bg-transparent text-zinc-500 hover:text-zinc-700",
                  )}
                >
                  {cycle === "monthly" ? (
                    "Monthly billing"
                  ) : (
                    <>
                      Yearly billing{" "}
                      <span className="ml-1.5 bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 bg-clip-text text-[11px] font-semibold text-transparent">
                        Save 15%
                      </span>
                    </>
                  )}
                </button>
              ))}
            </div>
          )}

          <div className="relative mt-2 w-full max-w-[75.625rem]">
            {isLoading ? (
              <div className="grid w-full grid-cols-1 gap-4 px-[1rem] md:grid-cols-3 md:px-0">
                <Skeleton className="h-[26rem] rounded-2xl" />
                <Skeleton className="h-[26rem] rounded-2xl" />
                <Skeleton className="h-[26rem] rounded-2xl" />
              </div>
            ) : plans.length === 0 ? (
              <p className="text-center text-sm text-zinc-500">
                Subscriptions are temporarily unavailable. Please try again
                shortly.
              </p>
            ) : (
              <div
                className={cn(
                  "grid w-full gap-4 px-[1rem] md:px-0",
                  plans.length === 1 && "grid-cols-1",
                  plans.length === 2 && "grid-cols-1 md:grid-cols-2",
                  plans.length >= 3 && "grid-cols-1 md:grid-cols-3",
                )}
              >
                {plans.map((plan) => (
                  <PlanCard
                    key={plan.key}
                    plan={plan}
                    country={country}
                    isYearly={isYearly}
                    onSelect={() => handleSelectPlan(plan.key)}
                    loading={isPending && selectedTier === plan.key}
                    disabled={isPending && selectedTier !== plan.key}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </Dialog.Content>
      {pendingTier && pendingTierLabel ? (
        <SwitchTierDialog
          isOpen={pendingTier !== null}
          onOpenChange={(open) => {
            if (!open) cancelPendingTier();
          }}
          targetTierLabel={pendingTierLabel}
          title={`Switch to ${pendingTierLabel}?`}
          confirmLabel={`Switch to ${pendingTierLabel}`}
          body="Your current Stripe subscription will be modified — you may be charged or refunded the prorated difference. Continue?"
          isSaving={isPending}
          onConfirm={() => {
            void confirmPendingTier();
          }}
        />
      ) : null}
    </Dialog>
  );
}
