"use client";

import { WelcomeStep } from "./components/WelcomeStep";
import { RoleStep } from "./components/RoleStep";
import { PainPointsStep } from "./components/PainPointsStep";
import { PreparingStep } from "./components/PreparingStep";
import { useOnboardingPage } from "./useOnboardingPage";

export default function OnboardingPage() {
  const { currentStep, isLoading, handlePreparingComplete } =
    useOnboardingPage();

  if (isLoading) return null;

  switch (currentStep) {
    case 1:
      return <WelcomeStep />;
    case 2:
      return <RoleStep />;
    case 3:
      return <PainPointsStep />;
    case 4:
      return <PreparingStep onComplete={handlePreparingComplete} />;
  }
}
