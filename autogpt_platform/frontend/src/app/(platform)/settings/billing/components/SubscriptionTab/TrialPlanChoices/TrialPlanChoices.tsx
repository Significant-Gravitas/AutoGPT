"use client";

import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";

import { ConfirmTrialPlanDialog } from "./ConfirmTrialPlanDialog";
import { ConfirmWithBankDialog } from "./ConfirmWithBankDialog";
import { formatChoicePrice } from "./helpers";
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
    confirmPlan,
    requestedTier,
    paymentPage,
    onSelectOwnPlan,
    onSelectUpgradePlan,
    onConfirmPlan,
    onCloseConfirm,
    onClosePaymentPage,
  } = useTrialPlanChoices(offer);

  if (!isVisible) return null;
  const isBusy = requestedTier !== null;
  const dialogPlan = confirmPlan ?? ownPlan;

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
      <ConfirmTrialPlanDialog
        isOpen={confirmPlan !== null}
        planLabel={dialogPlan.label}
        price={formatChoicePrice(dialogPlan)}
        isSaving={requestedTier === dialogPlan.tier}
        onConfirm={onConfirmPlan}
        onClose={onCloseConfirm}
      />
      <ConfirmWithBankDialog page={paymentPage} onClose={onClosePaymentPage} />
    </section>
  );
}
