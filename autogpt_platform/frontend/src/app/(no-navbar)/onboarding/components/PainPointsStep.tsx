"use client";

import {
  CaretLeft,
  MagnifyingGlass,
  EnvelopeSimple,
  ChartLineUp,
  Headset,
  DeviceMobile,
  Folder,
  CalendarBlank,
  Flask,
  PencilSimple,
} from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useOnboardingWizardStore } from "../store";
import { SelectableCard } from "./SelectableCard";
import { StepIndicator } from "./StepIndicator";

const PAIN_POINTS = [
  {
    id: "Finding leads",
    label: "Finding leads",
    icon: <MagnifyingGlass size={24} />,
  },
  {
    id: "Email & outreach",
    label: "Email & outreach",
    icon: <EnvelopeSimple size={24} />,
  },
  {
    id: "Reports & data",
    label: "Reports & data",
    icon: <ChartLineUp size={24} />,
  },
  {
    id: "Customer support",
    label: "Customer support",
    icon: <Headset size={24} />,
  },
  {
    id: "Social media",
    label: "Social media",
    icon: <DeviceMobile size={24} />,
  },
  {
    id: "CRM & data entry",
    label: "CRM & data entry",
    icon: <Folder size={24} />,
  },
  {
    id: "Scheduling",
    label: "Scheduling",
    icon: <CalendarBlank size={24} />,
  },
  { id: "Research", label: "Research", icon: <Flask size={24} /> },
  {
    id: "Something else",
    label: "Something else",
    icon: <PencilSimple size={24} />,
  },
] as const;

export function PainPointsStep() {
  const painPoints = useOnboardingWizardStore((s) => s.painPoints);
  const otherPainPoint = useOnboardingWizardStore((s) => s.otherPainPoint);
  const togglePainPoint = useOnboardingWizardStore((s) => s.togglePainPoint);
  const setOtherPainPoint = useOnboardingWizardStore(
    (s) => s.setOtherPainPoint,
  );
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const prevStep = useOnboardingWizardStore((s) => s.prevStep);

  const hasSomethingElse = painPoints.includes("Something else");
  const canContinue =
    painPoints.length > 0 && (!hasSomethingElse || otherPainPoint.trim());

  function handleLaunch() {
    if (canContinue) {
      nextStep();
    }
  }

  return (
    <div className="flex w-full max-w-lg flex-col items-center gap-6 px-4">
      <button
        type="button"
        onClick={prevStep}
        className="flex items-center gap-1 self-start text-sm text-muted-foreground hover:text-foreground"
      >
        <CaretLeft size={16} />
        Back
      </button>

      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          What&apos;s eating your time?
        </h1>
        <p className="text-sm text-muted-foreground">
          Pick the tasks you&apos;d love to hand off to AutoPilot
        </p>
      </div>

      <div className="grid w-full grid-cols-3 gap-3">
        {PAIN_POINTS.map((p) => (
          <SelectableCard
            key={p.id}
            icon={p.icon}
            label={p.label}
            selected={painPoints.includes(p.id)}
            onClick={() => togglePainPoint(p.id)}
          />
        ))}
      </div>

      {hasSomethingElse && (
        <Input
          placeholder="What else takes up your time?"
          value={otherPainPoint}
          onChange={(e) => setOtherPainPoint(e.target.value)}
          autoFocus
        />
      )}

      <p className="text-xs text-muted-foreground">
        Pick as many as you want — you can always change later
      </p>

      <Button
        onClick={handleLaunch}
        disabled={!canContinue}
        className="w-full max-w-xs"
      >
        Launch Autopilot
      </Button>

      <StepIndicator totalSteps={3} currentStep={3} />
    </div>
  );
}
