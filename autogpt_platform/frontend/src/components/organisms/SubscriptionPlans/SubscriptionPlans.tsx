"use client";

import type { ReactNode } from "react";
import { PlanDetailsDialog } from "./components/PlanDetailsDialog/PlanDetailsDialog";
import { PlanFooter } from "./components/PlanFooter";
import { PlanHeader } from "./components/PlanHeader";
import { SubscriptionOffer } from "./components/SubscriptionOffer";
import { DEFAULT_GOAL_SURFACE, type SubscriptionPlansProps } from "./helpers";
import { useSubscriptionPlans } from "./useSubscriptionPlans";

export function SubscriptionPlans({
  trialStatus,
  header,
  goalSurface = DEFAULT_GOAL_SURFACE,
  ...rest
}: SubscriptionPlansProps & {
  trialStatus?: ReactNode;
  header?: ReactNode;
}) {
  const { dialog, setDialog, setDialogOpen } = useSubscriptionPlans();
  const props = { ...rest, goalSurface };
  return (
    <div className="mx-auto w-full max-w-7xl px-5 font-sans sm:px-8">
      <section aria-label="Choose your AutoGPT plan">
        <PlanHeader {...props} header={header} />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {props.plans.map((plan) => (
            <SubscriptionOffer
              key={plan.key}
              {...props}
              plan={plan}
              onTrialDetails={() => setDialog("trial")}
            />
          ))}
        </div>
        {trialStatus && (
          <div className="mx-auto mt-4 w-full max-w-3xl empty:hidden">
            {trialStatus}
          </div>
        )}
        <PlanFooter onCompare={() => setDialog("compare")} />
      </section>
      <PlanDetailsDialog
        kind={dialog}
        setOpen={setDialogOpen}
        trialOffer={props.trialOffer}
        plans={props.plans}
      />
    </div>
  );
}
