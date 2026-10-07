"use client";

import { BillingOffer } from "./BillingOffer";
import { AutopilotUsageCard } from "./AutopilotUsageCard/AutopilotUsageCard";
import { InvoicesCard } from "./InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "./PaymentMethodCard/PaymentMethodCard";
import { YourPlanCard } from "./YourPlanCard/YourPlanCard";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";
import { useTrialStatus } from "@/services/trials/useTrialStatus";

export function SubscriptionTab() {
  const { data: trial, isLoading } = useTrialStatus();
  const showPlan = !isLoading && (!trial?.active || trial.converted);
  return (
    <div className="flex flex-col gap-4">
      <TrialCheckoutConfirmation />
      <TrialCard />
      {showPlan ? <YourPlanCard index={0} showUsage /> : null}
      {!showPlan && !isLoading ? (
        <div className="grid items-stretch gap-4 md:grid-cols-[1.15fr_1fr]">
          <AutopilotUsageCard index={1} />
          <BillingOffer />
        </div>
      ) : null}
      <PaymentMethodCard index={2} />
      <InvoicesCard index={3} />
    </div>
  );
}
