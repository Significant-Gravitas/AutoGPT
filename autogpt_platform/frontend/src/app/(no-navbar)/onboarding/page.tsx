"use client";

import { CaretLeft } from "@phosphor-icons/react";
import { ProgressBar } from "./components/ProgressBar";
import { StepIndicator } from "./components/StepIndicator";
import { PainPointsStep } from "./steps/PainPointsStep";
import { PreparingStep } from "./steps/PreparingStep";
import { RoleStep } from "./steps/RoleStep";
import { SubscriptionStep } from "./steps/SubscriptionStep/SubscriptionStep";
import { WelcomeStep } from "./steps/WelcomeStep";
import { useOnboardingWizardStore } from "./store";
import { useOnboardingPage } from "./useOnboardingPage";

export default function OnboardingPage() {
  const {
    currentStep,
    isLoading,
    handlePreparingComplete,
    isPaymentEnabled,
    preparingStep,
    totalSteps,
  } = useOnboardingPage();
  const prevStep = useOnboardingWizardStore((s) => s.prevStep);

  if (isLoading) return null;

  // ProgressBar + StepIndicator track only the user-interactive steps.
  // PreparingStep is a transition view that hides both indicators.
  const showDots = currentStep <= totalSteps;
  const showBack = currentStep > 1 && currentStep <= totalSteps;
  const showProgressBar = currentStep <= totalSteps;

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
          <CaretLeft size={16} />
          Back
        </button>
      )}

      <div className="flex flex-1 items-center py-16">
        {currentStep === 1 && <WelcomeStep />}
        {currentStep === 2 && <RoleStep />}
        {currentStep === 3 && <PainPointsStep />}
        {isPaymentEnabled && currentStep === 4 && <SubscriptionStep />}
        {currentStep === preparingStep && (
          <PreparingStep onComplete={handlePreparingComplete} />
        )}
      </div>

      {showDots && (
        <div className="pb-8">
          <StepIndicator totalSteps={totalSteps} currentStep={currentStep} />
        </div>
      )}
    </div>
  );
}
