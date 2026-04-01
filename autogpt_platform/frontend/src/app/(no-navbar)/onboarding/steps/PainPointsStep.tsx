"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  CalendarBlank,
  ChartLineUp,
  DeviceMobile,
  EnvelopeSimple,
  Flask,
  Folder,
  Headset,
  MagnifyingGlass,
  PencilSimple,
} from "@phosphor-icons/react";

import { FadeIn } from "../components/FadeIn";
import { SelectableCard } from "../components/SelectableCard";
import { useOnboardingWizardStore } from "../store";

const PAIN_POINTS = [
  {
    id: "Finding leads",
    label: "Finding leads",
    icon: <MagnifyingGlass size={32} />,
  },
  {
    id: "Email & outreach",
    label: "Email & outreach",
    icon: <EnvelopeSimple size={32} />,
  },
  {
    id: "Reports & data",
    label: "Reports & data",
    icon: <ChartLineUp size={32} />,
  },
  {
    id: "Customer support",
    label: "Customer support",
    icon: <Headset size={32} />,
  },
  {
    id: "Social media",
    label: "Social media",
    icon: <DeviceMobile size={32} />,
  },
  {
    id: "CRM & data entry",
    label: "CRM & data entry",
    icon: <Folder size={32} />,
  },
  {
    id: "Scheduling",
    label: "Scheduling",
    icon: <CalendarBlank size={32} />,
  },
  {
    id: "Research",
    label: "Research",
    icon: <Flask size={32} />,
  },
  {
    id: "Something else",
    label: "Something else",
    icon: <PencilSimple size={32} />,
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

  const hasSomethingElse = painPoints.includes("Something else");
  const canContinue =
    painPoints.length > 0 && (!hasSomethingElse || otherPainPoint.trim());

  function handleLaunch() {
    if (canContinue) {
      nextStep();
    }
  }

  return (
    <FadeIn>
      <div className="flex w-full flex-col items-center gap-12 px-4">
        <div className="flex max-w-lg flex-col items-center gap-2 px-4 text-center">
          <Text
            variant="h3"
            className="!text-[1.5rem] !leading-[2rem] md:!text-[1.75rem] md:!leading-[2.5rem]"
          >
            What&apos;s eating your time?
          </Text>
          <Text variant="lead" className="!text-zinc-500">
            Pick the tasks you&apos;d love to hand off to AutoPilot
          </Text>
        </div>

        <div className="flex w-full flex-col items-center gap-4">
          <div className="flex w-full max-w-[100vw] flex-nowrap gap-4 overflow-x-auto px-8 scrollbar-none md:grid md:grid-cols-3 md:overflow-hidden md:px-0">
            {PAIN_POINTS.map((p) => (
              <SelectableCard
                key={p.id}
                icon={p.icon}
                label={p.label}
                selected={painPoints.includes(p.id)}
                onClick={() => togglePainPoint(p.id)}
                className="p-8"
              />
            ))}
          </div>
          {!hasSomethingElse ? (
            <Text variant="small" className="!text-zinc-500">
              Pick as many as you want — you can always change later
            </Text>
          ) : null}
        </div>

        {hasSomethingElse && (
          <div className="-mb-5 w-full px-8 md:px-0">
            <Input
              id="other-pain-point"
              label="Other pain point"
              hideLabel
              placeholder="What else takes up your time?"
              value={otherPainPoint}
              onChange={(e) => setOtherPainPoint(e.target.value)}
              autoFocus
            />
          </div>
        )}

        <Button
          onClick={handleLaunch}
          disabled={!canContinue}
          className="w-full max-w-xs"
        >
          Launch Autopilot
        </Button>
      </div>
    </FadeIn>
  );
}
