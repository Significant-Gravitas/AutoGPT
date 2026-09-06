"use client";

import { AutopilotUsageCard } from "./AutopilotUsageCard/AutopilotUsageCard";
import { InvoicesCard } from "./InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "./PaymentMethodCard/PaymentMethodCard";
import { YourPlanCard } from "./YourPlanCard/YourPlanCard";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";
import { useTrialStatus } from "@/services/trials/useTrialStatus";

export function SubscriptionTab() {
  const { data: trial } = useTrialStatus();
  const showPlan =
    !trial?.status ||
    trial.converted ||
    trial.status === "canceled" ||
    trial.status === "checkout_pending";
  return (
    <div className="flex flex-col gap-6">
      <TrialCheckoutConfirmation />
      <TrialCard />
      {showPlan ? <YourPlanCard index={0} /> : null}
      <AutopilotUsageCard index={1} />
      <PaymentMethodCard index={2} />
      <InvoicesCard index={3} />
    </div>
  );
}
