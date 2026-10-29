"use client";

import { Text } from "@/components/atoms/Text/Text";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";
import { PaymentRecoveryCard } from "@/components/organisms/UsageExperience/PaymentRecoveryCard";

import { BillingOffer } from "./BillingOffer";
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
  const {
    showPlan,
    showTrialCard,
    planChoicesOffer,
    isConfirmingPlan,
    isLoading,
    hasPlanChoices,
    paymentFailed,
    payment,
  } = useSubscriptionTab(isPlanCheckoutReturn);
  return (
    <div className="flex flex-col gap-4">
      <TrialCheckoutConfirmation />
      <div className="flex flex-col gap-3 empty:hidden">
        {paymentFailed ? (
          <PaymentRecoveryCard
            payment={payment}
            fallback="Update your payment method below to continue. Your chats, workflows and results are saved."
          />
        ) : null}
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
      {showPlan ? <YourPlanCard index={0} showUsage /> : null}
      {!showPlan && !isLoading && !paymentFailed ? (
        <div
          className={
            hasPlanChoices
              ? "grid gap-4"
              : "grid items-stretch gap-4 md:grid-cols-[1.15fr_1fr]"
          }
        >
          <AutopilotUsageCard index={1} />
          {!hasPlanChoices ? <BillingOffer /> : null}
        </div>
      ) : null}
      <PaymentMethodCard index={2} />
      <InvoicesCard index={3} />
    </div>
  );
}
