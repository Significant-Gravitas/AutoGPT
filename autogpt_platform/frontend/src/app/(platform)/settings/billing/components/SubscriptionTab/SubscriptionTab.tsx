"use client";

import { AutopilotUsageCard } from "./AutopilotUsageCard/AutopilotUsageCard";
import { InvoicesCard } from "./InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "./PaymentMethodCard/PaymentMethodCard";
import { getCancelPendingOffer } from "./TrialPlanChoices/helpers";
import { TrialPlanChoices } from "./TrialPlanChoices/TrialPlanChoices";
import { YourPlanCard } from "./YourPlanCard/YourPlanCard";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";
import { useTrialStatus } from "@/services/trials/useTrialStatus";

export function SubscriptionTab() {
  const { data: trial, isLoading } = useTrialStatus();
  const showPlan = !isLoading && (!trial?.active || trial.converted);
  const cancelPendingOffer = getCancelPendingOffer(trial);
  return (
    <div className="flex flex-col gap-6">
      <TrialCheckoutConfirmation />
      <div className="flex flex-col gap-3 empty:hidden">
        <TrialCard />
        {cancelPendingOffer ? (
          <TrialPlanChoices offer={cancelPendingOffer} />
        ) : null}
      </div>
      {showPlan ? <YourPlanCard index={0} /> : null}
      <AutopilotUsageCard index={1} />
      <PaymentMethodCard index={2} />
      <InvoicesCard index={3} />
    </div>
  );
}
