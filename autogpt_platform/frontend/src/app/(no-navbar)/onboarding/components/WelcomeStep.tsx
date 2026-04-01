"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useOnboardingWizardStore } from "../store";
import { StepIndicator } from "./StepIndicator";

export function WelcomeStep() {
  const name = useOnboardingWizardStore((s) => s.name);
  const setName = useOnboardingWizardStore((s) => s.setName);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (name.trim()) {
      nextStep();
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full max-w-md flex-col items-center gap-8 px-4"
    >
      <div className="flex flex-col items-center gap-3 text-center">
        <h1 className="text-3xl font-semibold tracking-tight">
          Welcome to AutoGPT
        </h1>
        <p className="text-muted-foreground">
          Let&apos;s personalize your experience so Autopilot can start saving
          you time right away
        </p>
      </div>

      <div className="flex w-full flex-col gap-2">
        <label htmlFor="first-name" className="text-sm font-medium">
          Your first name
        </label>
        <Input
          id="first-name"
          placeholder="e.g. John"
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoFocus
        />
      </div>

      <Button type="submit" disabled={!name.trim()} className="w-full max-w-xs">
        Continue
      </Button>

      <StepIndicator totalSteps={3} currentStep={1} />
    </form>
  );
}
