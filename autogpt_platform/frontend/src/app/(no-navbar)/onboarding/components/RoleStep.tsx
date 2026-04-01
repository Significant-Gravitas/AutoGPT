"use client";

import {
  CaretLeft,
  Gear,
  Wrench,
  ChartBar,
  Megaphone,
  Hammer,
  Code,
  Users,
  PencilSimple,
} from "@phosphor-icons/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useOnboardingWizardStore } from "../store";
import { SelectableCard } from "./SelectableCard";
import { StepIndicator } from "./StepIndicator";

const ROLES = [
  { id: "Founder/CEO", label: "Founder / CEO", icon: <Gear size={24} /> },
  { id: "Operations", label: "Operations", icon: <Wrench size={24} /> },
  { id: "Sales/BD", label: "Sales / BD", icon: <ChartBar size={24} /> },
  { id: "Marketing", label: "Marketing", icon: <Megaphone size={24} /> },
  { id: "Product/PM", label: "Product / PM", icon: <Hammer size={24} /> },
  { id: "Engineering", label: "Engineering", icon: <Code size={24} /> },
  { id: "HR/People", label: "HR / People", icon: <Users size={24} /> },
  {
    id: "Other",
    label: "Other",
    icon: <PencilSimple size={24} />,
  },
] as const;

export function RoleStep() {
  const name = useOnboardingWizardStore((s) => s.name);
  const role = useOnboardingWizardStore((s) => s.role);
  const otherRole = useOnboardingWizardStore((s) => s.otherRole);
  const setRole = useOnboardingWizardStore((s) => s.setRole);
  const setOtherRole = useOnboardingWizardStore((s) => s.setOtherRole);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const prevStep = useOnboardingWizardStore((s) => s.prevStep);

  const isOther = role === "Other";
  const canContinue = role && (!isOther || otherRole.trim());

  function handleContinue() {
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
          What best describes you, {name}?
        </h1>
        <p className="text-sm text-muted-foreground">
          AutoPilot will tailor automations to your world
        </p>
      </div>

      <div className="grid w-full grid-cols-4 gap-3">
        {ROLES.map((r) => (
          <SelectableCard
            key={r.id}
            icon={r.icon}
            label={r.label}
            selected={role === r.id}
            onClick={() => setRole(r.id)}
          />
        ))}
      </div>

      {isOther && (
        <Input
          placeholder="Describe your role..."
          value={otherRole}
          onChange={(e) => setOtherRole(e.target.value)}
          autoFocus
        />
      )}

      <Button
        onClick={handleContinue}
        disabled={!canContinue}
        className="w-full max-w-xs"
      >
        Continue
      </Button>

      <StepIndicator totalSteps={3} currentStep={2} />
    </div>
  );
}
