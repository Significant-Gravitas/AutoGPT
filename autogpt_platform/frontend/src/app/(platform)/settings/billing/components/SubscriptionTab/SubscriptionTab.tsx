"use client";

import { Text } from "@/components/atoms/Text/Text";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";

import { AutopilotUsageCard } from "./AutopilotUsageCard/AutopilotUsageCard";
import { InvoicesCard } from "./InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "./PaymentMethodCard/PaymentMethodCard";
import { TrialPlanChoices } from "./TrialPlanChoices/TrialPlanChoices";
import { useSubscriptionTab } from "./useSubscriptionTab";
import { YourPlanCard } from "./YourPlanCard/YourPlanCard";

interface Props {
  isPlanCheckoutReturn?: boolean;
}

export function SubscriptionTab({ isPlanCheckoutReturn = false }: Props) {
  const { showPlan, showTrialCard, planChoicesOffer, isConfirmingPlan } =
    useSubscriptionTab(isPlanCheckoutReturn);
  return (
    <div className="flex flex-col gap-6">
      <TrialCheckoutConfirmation />
      <div className="flex flex-col gap-3 empty:hidden">
        {showTrialCard ? <TrialCard /> : null}
        {isConfirmingPlan ? (
          <Text variant="small" tone="secondary" role="status" className="px-4">
            Finishing your new plan…
          </Text>
        ) : null}
        {planChoicesOffer ? (
          <TrialPlanChoices offer={planChoicesOffer} />
        ) : null}
      </div>
      {showPlan ? <YourPlanCard index={0} /> : null}
      <AutopilotUsageCard index={1} />
      <PaymentMethodCard index={2} />
      <InvoicesCard index={3} />
    </div>
  );
}
