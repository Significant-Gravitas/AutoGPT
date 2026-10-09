"use client";

import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";

import { ConfirmTrialPlanDialog } from "./ConfirmTrialPlanDialog";
import { PlanChoice } from "./PlanChoice";
import { useTrialPlanChoices } from "./useTrialPlanChoices";

interface Props {
  offer: TrialOfferResponse;
}

export function TrialPlanChoices({ offer }: Props) {
  const {
    isVisible,
    ownPlan,
    upgradePlan,
    ownPrice,
    requestedTier,
    isConfirmOpen,
    error,
    onSelectOwnPlan,
    onConfirmOwnPlan,
    onCloseConfirm,
    onSelectUpgradePlan,
  } = useTrialPlanChoices(offer);

  if (!isVisible) return null;
  const isBusy = requestedTier !== null;

  return (
    <section aria-label="Plan choices" className="flex flex-col gap-3">
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <PlanChoice
          plan={ownPlan}
          actionLabel={`Subscribe to ${ownPlan.label}`}
          variant="primary"
          isLoading={requestedTier === ownPlan.tier}
          isDisabled={isBusy}
          onSelect={onSelectOwnPlan}
        />
        {upgradePlan ? (
          <PlanChoice
            plan={upgradePlan}
            actionLabel={`Upgrade to ${upgradePlan.label}`}
            variant="outline"
            isLoading={requestedTier === upgradePlan.tier}
            isDisabled={isBusy}
            onSelect={onSelectUpgradePlan}
          />
        ) : null}
      </ul>
      {error ? (
        <p role="alert" className="px-1 text-sm text-destructive">
          {error}
        </p>
      ) : null}
      <ConfirmTrialPlanDialog
        isOpen={isConfirmOpen}
        planLabel={ownPlan.label}
        price={ownPrice}
        isSaving={requestedTier === ownPlan.tier}
        onConfirm={onConfirmOwnPlan}
        onClose={onCloseConfirm}
      />
    </section>
  );
}
