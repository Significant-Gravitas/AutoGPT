"use client";

import { AutopilotUsageCard } from "./AutopilotUsageCard/AutopilotUsageCard";
import { InvoicesCard } from "./InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "./PaymentMethodCard/PaymentMethodCard";
import { YourPlanCard } from "./YourPlanCard/YourPlanCard";

export function SubscriptionTab() {
  return (
    <div className="flex flex-col gap-6">
      <YourPlanCard index={0} />
      <AutopilotUsageCard index={1} />
      <PaymentMethodCard index={2} />
      <InvoicesCard index={3} />
    </div>
  );
}
