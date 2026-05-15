"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { SelectableCard } from "../components/SelectableCard";
import { usePainPointsStep } from "./usePainPointsStep";
import { Emoji } from "@/components/atoms/Emoji/Emoji";

const ALL_PAIN_POINTS: { id: string; label: string; icon: ReactNode }[] = [
  {
    id: "Finding leads",
    label: "Finding leads",
    icon: <Emoji text="🔍" size={32} />,
  },
  {
    id: "Email & outreach",
    label: "Email & outreach",
    icon: <Emoji text="📧" size={32} />,
  },
  {
    id: "Reports & data",
    label: "Reports & data",
    icon: <Emoji text="📊" size={32} />,
  },
  {
    id: "Customer support",
    label: "Customer support",
    icon: <Emoji text="💬" size={32} />,
  },
  {
    id: "Social media",
    label: "Social media",
    icon: <Emoji text="📱" size={32} />,
  },
  {
    id: "CRM & data entry",
    label: "CRM & data entry",
    icon: <Emoji text="📝" size={32} />,
  },
  {
    id: "Scheduling",
    label: "Scheduling",
    icon: <Emoji text="🗓️" size={32} />,
  },
  { id: "Research", label: "Research", icon: <Emoji text="🔬" size={32} /> },
  {
    id: "Something else",
    label: "Something else",
    icon: <Emoji text="🚩" size={32} />,
  },
];

function orderPainPoints(topIDs: string[]) {
  const top = topIDs
    .map((id) => ALL_PAIN_POINTS.find((p) => p.id === id))
    .filter((p): p is (typeof ALL_PAIN_POINTS)[number] => p != null);
  const rest = ALL_PAIN_POINTS.filter(
    (p) => !topIDs.includes(p.id) && p.id !== "Something else",
  );
  const somethingElse = ALL_PAIN_POINTS.find((p) => p.id === "Something else")!;
  return [...top, ...rest, somethingElse];
}

export function PainPointsStep() {
  const {
    topIDs,
    painPoints,
    otherPainPoint,
    togglePainPoint,
    setOtherPainPoint,
    hasSomethingElse,
    atLimit,
    shaking,
    canContinue,
    handleLaunch,
  } = usePainPointsStep();

  const orderedPainPoints = orderPainPoints(topIDs);

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
            {orderedPainPoints.map((p) => (
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
          <Text
            variant="small"
            className={cn(
              "transition-colors",
              atLimit && canContinue ? "!text-green-600" : "!text-zinc-500",
              shaking && "animate-shake",
            )}
          >
            {shaking
              ? "You've picked 3 — tap one to swap it out"
              : atLimit && canContinue
                ? "3 selected — you're all set!"
                : atLimit && hasSomethingElse
                  ? "Tell us what else takes up your time"
                  : "Pick up to 3 to start — AutoPilot can help with anything else later"}
          </Text>
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
          Continue
        </Button>
      </div>
    </FadeIn>
  );
}
