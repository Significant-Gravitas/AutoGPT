"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { SubscriptionPlans } from "@/components/organisms/SubscriptionPlans/SubscriptionPlans";
import { TrialCardContent } from "@/components/organisms/TrialCard/TrialCard";
import { Logout03Icon } from "@hugeicons/core-free-icons";
import { SwitchTierDialog } from "../settings/billing/components/SubscriptionTab/YourPlanCard/SwitchTierDialog";
import { PaywallPlanChoices } from "./components/PaywallPlanChoices";
import { PaywallHeader } from "./components/PaywallHeader";
import { usePaywallModal } from "./usePaywallModal";

export function PaywallModal() {
  const {
    isLoading,
    plans,
    retryLoadPlans,
    isRetryingPlans,
    country,
    selectedCycle,
    setSelectedCycle,
    handleSelectPlan,
    isPending,
    selectedTier,
    pendingTier,
    pendingTierLabel,
    confirmPendingTier,
    cancelPendingTier,
    handleLogout,
    handleClose,
    trial,
    trialOffer,
  } = usePaywallModal();

  return (
    <Dialog
      variant="compact"
      title="Choose your plan"
      styling={{ maxWidth: trialOffer ? "1120px" : "700px" }}
      controlled={{
        isOpen: true,
        set: (open) => {
          if (!open) handleClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="relative flex w-full flex-col items-center gap-4 px-2 py-2">
          <div className="flex w-full justify-start">
            <Button
              variant="ghost"
              size="small"
              onClick={handleLogout}
              leftIcon={<Icon icon={Logout03Icon} size={16} />}
              className="bg-white/90 text-zinc-500 hover:text-zinc-700"
            >
              Log out
            </Button>
          </div>
          <PaywallHeader />

          <div className="relative mt-2 w-full">
            {isLoading || trial.isLoading ? (
              <div className="grid w-full grid-cols-1 gap-4 px-[1rem] md:grid-cols-3 md:px-0">
                <Skeleton className="h-[26rem] rounded-2xl" />
                <Skeleton className="h-[26rem] rounded-2xl" />
                <Skeleton className="h-[26rem] rounded-2xl" />
              </div>
            ) : plans.length === 0 ? (
              <div className="flex flex-col items-center gap-3">
                <p className="text-center text-sm text-zinc-500">
                  Subscriptions are temporarily unavailable. Please try again
                  shortly.
                </p>
                <Button
                  variant="secondary"
                  size="small"
                  onClick={retryLoadPlans}
                  loading={isRetryingPlans}
                >
                  Retry
                </Button>
              </div>
            ) : !trialOffer ? (
              <div className="space-y-4">
                <PaywallPlanChoices
                  plans={plans}
                  cycle={selectedCycle}
                  onCycle={setSelectedCycle}
                  onSelect={handleSelectPlan}
                  pending={isPending}
                  selectedTier={selectedTier}
                />
                {!trial.trial?.eligible && (
                  <TrialCardContent returnTo="billing" controller={trial} />
                )}
              </div>
            ) : (
              <SubscriptionPlans
                // The modal renders its own title above, so the organism
                // contributes only the billing toggle.
                header={null}
                goalSurface="upgrade_modal"
                plans={plans}
                country={country}
                billing={selectedCycle}
                onBillingChange={setSelectedCycle}
                onSelectPlan={handleSelectPlan}
                isUpdatingTier={isPending}
                selectedPlan={selectedTier}
                trialOffer={trialOffer}
                onStartTrial={trial.startTrial}
                isStartingTrial={trial.isStarting}
                trialError={trial.error}
                trialStatus={
                  !trialOffer && (
                    <TrialCardContent returnTo="billing" controller={trial} />
                  )
                }
              />
            )}
          </div>
        </div>
        <Button variant="ghost" className="mt-3 w-full" onClick={handleClose}>
          Back to billing
        </Button>
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
