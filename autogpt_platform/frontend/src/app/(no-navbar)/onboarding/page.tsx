"use client";
import { ProgressBar } from "./components/ProgressBar";
import { StepIndicator } from "./components/StepIndicator";
import { BrainDumpStep } from "./steps/BrainDumpStep/BrainDumpStep";
import { PainPointsStep } from "./steps/PainPointsStep";
import { PreparingStep } from "./steps/PreparingStep";
import { RoleStep } from "./steps/RoleStep";
import { ConnectStep } from "./steps/ConnectStep/ConnectStep";
import { SubscriptionStep } from "./steps/SubscriptionStep/SubscriptionStep";
import { IntroStep } from "./steps/IntroStep/IntroStep";
import { HireStep } from "./steps/HireStep/HireStep";
import { useOnboardingWizardStore } from "./store";
import { useOnboardingPage } from "./useOnboardingPage";
import { ArrowLeft01Icon, Logout03Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

export default function OnboardingPage() {
  const {
    currentStep,
    isLoading,
    handlePreparingComplete,
    isPaymentEnabled,
    isSelfHostConnectEnabled,
    isBrainDumpEnabled,
    steps,
    preparingStep,
    totalSteps,
    trialConfirmation,
  } = useOnboardingPage();
  const prevStep = useOnboardingWizardStore((s) => s.prevStep);
  const isStepBusy = useOnboardingWizardStore((s) => s.isStepBusy);

  if (isLoading)
    return !trialConfirmation.ready ? (
      <Text variant="body" role="status">
        Confirming your trial and card setup…
      </Text>
    ) : null;

  // ProgressBar + StepIndicator track only the user-interactive steps.
  // PreparingStep is a transition view that hides both indicators.
  const showDots = currentStep <= totalSteps;
  // Back is hidden on the first step and while the current step is
  // mid-flight (brain dump processing) — there is nothing coherent to go
  // back to.
  const firstContentStep = isPaymentEnabled ? 2 : 1;
  const showBack =
    currentStep > firstContentStep && currentStep <= totalSteps && !isStepBusy;
  const showProgressBar = currentStep <= totalSteps;
  const showLogout = currentStep <= totalSteps;

  return (
    <div className="flex min-h-screen w-full flex-col items-center">
      {trialConfirmation.error ? (
        <ErrorCard
          context="your trial"
          responseError={{ message: trialConfirmation.error }}
          onRetry={trialConfirmation.retry}
        />
      ) : null}
      {showProgressBar && (
        <ProgressBar currentStep={currentStep} totalSteps={totalSteps} />
      )}

      {showBack && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={prevStep}
          leadingIcon={ArrowLeft01Icon}
          className="absolute left-6 top-6 text-zinc-500 hover:text-zinc-900"
        >
          Back
        </Button>
      )}

      <div className="flex w-full min-w-0 flex-1 items-center justify-center pb-8 pt-16">
        {currentStep === steps.team && <IntroStep slide="team" />}
        {currentStep === steps.autopilot && <IntroStep slide="autopilot" />}
        {currentStep === steps.role && <RoleStep />}
        {currentStep === steps.painPoints &&
          (isBrainDumpEnabled ? <BrainDumpStep /> : <PainPointsStep />)}
        {currentStep === steps.hire && <HireStep />}
        {isSelfHostConnectEnabled && currentStep === steps.connect && (
          <ConnectStep />
        )}
        {isPaymentEnabled && currentStep === steps.subscription && (
          <SubscriptionStep />
        )}
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
        <Button
          as="NextLink"
          href="/logout"
          variant="ghost"
          size="xs"
          leadingIcon={Logout03Icon}
          className="absolute bottom-6 left-6 text-zinc-500 hover:text-zinc-900"
        >
          Log out
        </Button>
      )}
    </div>
  );
}
