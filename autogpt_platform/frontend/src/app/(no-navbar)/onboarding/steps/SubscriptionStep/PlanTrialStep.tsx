"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { TrialCardContent } from "@/components/organisms/TrialCard/TrialCard";
import { useOnboardingWizardStore } from "../../store";
import { PLAN_HEADING_ID, PlanPanel } from "./components/PlanPanel/PlanPanel";
import { usePlan } from "./components/PlanPanel/usePlan";
import { PlanStepCard } from "./components/PlanStepCard/PlanStepCard";
import { PaidPanel } from "./components/TrialPanel/PaidPanel";
import { TRIAL_TITLE_ID, TrialPanel } from "./components/TrialPanel/TrialPanel";
import { usePaywallStep } from "./usePaywallStep";

interface Props {
  hasPaywall: boolean;
}

// The step after the brain dump: Otto's plan, and the trial that starts it.
// Without a paywall (self-hosted, e2e) the plan stands alone with Continue.
export function PlanTrialStep({ hasPaywall }: Props) {
  return hasPaywall ? <PaywallPlanStep /> : <PlanOnlyStep />;
}

function PaywallPlanStep() {
  const { plan } = usePlan();
  const paywall = usePaywallStep();
  return (
    <PlanStepCard
      labelledBy={paywall.isLoading ? PLAN_HEADING_ID : TRIAL_TITLE_ID}
      twoColumns
    >
      <PlanPanel
        plan={plan}
        isPageTitle={paywall.isLoading}
        moment={paywall.isLoading || paywall.offer ? "trial" : "subscribe"}
      />
      {paywall.isLoading ? (
        <div className="flex flex-col gap-4 bg-surface p-8" aria-busy>
          <Skeleton className="h-8 w-2/3 rounded-lg" />
          <Skeleton className="h-64 rounded-2xl" />
          <Skeleton className="h-12 rounded-full" />
        </div>
      ) : paywall.offer ? (
        <TrialPanel
          offer={paywall.offer}
          isStarting={paywall.isStarting}
          isBlocked={paywall.isBlocked}
          error={paywall.error}
          onStart={paywall.startTrial}
        />
      ) : (
        <PaidPanel
          price={paywall.price}
          cadence={paywall.cadence}
          isProcessing={paywall.isProcessingCheckout}
          isBlocked={paywall.isUpdatingTier}
          onContinue={paywall.continueToCheckout}
          notice={
            <TrialCardContent
              returnTo="onboarding"
              controller={paywall.trialController}
            />
          }
        />
      )}
    </PlanStepCard>
  );
}

function PlanOnlyStep() {
  const { plan } = usePlan();
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  return (
    <PlanStepCard labelledBy={PLAN_HEADING_ID} twoColumns={false}>
      <PlanPanel
        plan={plan}
        moment="continue"
        standalone
        footer={
          <Button
            variant="primary"
            size="large"
            onClick={nextStep}
            textSize="body-md"
            className="mt-1 h-12 w-full"
          >
            Continue
          </Button>
        }
      />
    </PlanStepCard>
  );
}
