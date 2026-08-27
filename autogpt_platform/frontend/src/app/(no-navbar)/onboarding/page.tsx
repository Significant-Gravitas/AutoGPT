"use client";
import Link from "next/link";
import { ProgressBar } from "./components/ProgressBar";
import { StepIndicator } from "./components/StepIndicator";
import { BrainDumpStep } from "./steps/BrainDumpStep/BrainDumpStep";
import { PainPointsStep } from "./steps/PainPointsStep";
import { PreparingStep } from "./steps/PreparingStep";
import { RoleStep } from "./steps/RoleStep";
import { SubscriptionStep } from "./steps/SubscriptionStep/SubscriptionStep";
import { WelcomeStep } from "./steps/WelcomeStep";
import { PAYWALL_FIRST_STEPS, useOnboardingWizardStore } from "./store";
import { useOnboardingPage } from "./useOnboardingPage";
import { ArrowLeft01Icon, Logout03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export default function OnboardingPage() {
  const {
    currentStep,
    isLoading,
    handlePreparingComplete,
    isPaymentEnabled,
    isBrainDumpEnabled,
    steps,
    preparingStep,
    totalSteps,
  } = useOnboardingPage();
  const prevStep = useOnboardingWizardStore((s) => s.prevStep);
  const isStepBusy = useOnboardingWizardStore((s) => s.isStepBusy);

  if (isLoading) return null;

  // ProgressBar + StepIndicator track only the user-interactive steps.
  // PreparingStep is a transition view that hides both indicators.
  const showDots = currentStep <= totalSteps;
  // Back is hidden on Welcome (the first profile step): going back from there
  // when payments are on would return the user to the paywall they already
  // paid through and let them re-trigger checkout. Also hidden while the
  // current step is mid-flight (brain dump processing) — there is nothing
  // coherent to go back to.
  const showBack =
    currentStep > steps.welcome && currentStep <= totalSteps && !isStepBusy;
  const showProgressBar = currentStep <= totalSteps;
  const showLogout = currentStep <= totalSteps;

  return (
    <div className="flex min-h-screen w-full flex-col items-center">
      {showProgressBar && (
        <ProgressBar currentStep={currentStep} totalSteps={totalSteps} />
      )}

      {showBack && (
        <button
          type="button"
          onClick={prevStep}
          className="text-md absolute left-6 top-6 flex items-center gap-1 text-zinc-500 transition-colors duration-200 hover:text-zinc-900"
        >
          <Icon icon={ArrowLeft01Icon} size={16} />
          Back
        </button>
      )}

      <div className="flex flex-1 items-center pb-8 pt-16">
        {isPaymentEnabled &&
          currentStep === PAYWALL_FIRST_STEPS.subscription && (
            <SubscriptionStep />
          )}
        {currentStep === steps.welcome && <WelcomeStep />}
        {currentStep === steps.role && <RoleStep />}
        {currentStep === steps.painPoints &&
          (isBrainDumpEnabled ? <BrainDumpStep /> : <PainPointsStep />)}
        {currentStep === preparingStep && (
          <PreparingStep
            onComplete={handlePreparingComplete}
            isBrainDumpEnabled={isBrainDumpEnabled}
          />
        )}
      </div>

      {showDots && (
        <div className="pb-8">
          <StepIndicator totalSteps={totalSteps} currentStep={currentStep} />
        </div>
      )}

      {showLogout && (
        <Link
          href="/logout"
          className="text-md absolute bottom-6 left-6 flex items-center gap-1 text-zinc-500 transition-colors duration-200 hover:text-zinc-900"
        >
          <Icon icon={Logout03Icon} size={16} />
          Log out
        </Link>
      )}
    </div>
  );
}
