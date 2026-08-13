"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { SelectableCard } from "../components/SelectableCard";
import { useOnboardingWizardStore } from "../store";
import { Emoji } from "@/components/atoms/Emoji/Emoji";
import { useEffect, useRef } from "react";

const IMG_SIZE = 42;

const ROLES = [
  {
    id: "Founder/CEO",
    label: "Founder / CEO",
    icon: <Emoji text="🎯" size={IMG_SIZE} />,
  },
  {
    id: "Operations",
    label: "Operations",
    icon: <Emoji text="⚙️" size={IMG_SIZE} />,
  },
  {
    id: "Sales/BD",
    label: "Sales / BD",
    icon: <Emoji text="📈" size={IMG_SIZE} />,
  },
  {
    id: "Marketing",
    label: "Marketing",
    icon: <Emoji text="📢" size={IMG_SIZE} />,
  },
  {
    id: "Product/PM",
    label: "Product / PM",
    icon: <Emoji text="🔨" size={IMG_SIZE} />,
  },
  {
    id: "Engineering",
    label: "Engineering",
    icon: <Emoji text="💻" size={IMG_SIZE} />,
  },
  {
    id: "HR/People",
    label: "HR / People",
    icon: <Emoji text="👤" size={IMG_SIZE} />,
  },
  { id: "Other", label: "Other", icon: <Emoji text="🚩" size={IMG_SIZE} /> },
] as const;

export function RoleStep() {
  const name = useOnboardingWizardStore((s) => s.name);
  const role = useOnboardingWizardStore((s) => s.role);
  const otherRole = useOnboardingWizardStore((s) => s.otherRole);
  const setRole = useOnboardingWizardStore((s) => s.setRole);
  const setOtherRole = useOnboardingWizardStore((s) => s.setOtherRole);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const autoAdvanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isOther = role === "Other";

  useEffect(() => {
    return () => {
      if (autoAdvanceTimer.current) clearTimeout(autoAdvanceTimer.current);
    };
  }, []);

  function handleRoleSelect(id: string) {
    if (autoAdvanceTimer.current) clearTimeout(autoAdvanceTimer.current);
    setRole(id);
    if (id !== "Other") {
      autoAdvanceTimer.current = setTimeout(nextStep, 350);
    }
  }

  function handleOtherContinue() {
    if (otherRole.trim()) {
      nextStep();
    }
  }

  return (
    <FadeIn>
      <div className="flex w-full flex-col items-center gap-12 px-4">
        <div className="mx-auto flex w-full max-w-lg flex-col items-center gap-2 px-4 text-center">
          <Text
            variant="h3"
            className="!text-[1.5rem] !leading-[2rem] md:!text-[1.75rem] md:!leading-[2.5rem]"
          >
            What best describes you, {name}?
          </Text>
          <Text variant="lead" className="!text-zinc-500">
            So AutoPilot knows how to help you best
          </Text>
        </div>

        <div className="flex w-full max-w-[100vw] flex-nowrap gap-4 overflow-x-auto px-8 scrollbar-none md:grid md:grid-cols-4 md:overflow-hidden md:px-0">
          {ROLES.map((r) => (
            <SelectableCard
              key={r.id}
              icon={r.icon}
              label={r.label}
              selected={role === r.id}
              onClick={() => handleRoleSelect(r.id)}
              className="p-8"
            />
          ))}
        </div>

        {isOther && (
          <>
            <div className="-mb-5 w-full px-8 md:px-0">
              <Input
                id="other-role"
                label="Other role"
                hideLabel
                placeholder="Describe your role..."
                value={otherRole}
                onChange={(e) => setOtherRole(e.target.value)}
                autoFocus
              />
            </div>

            <Button
              onClick={handleOtherContinue}
              disabled={!otherRole.trim()}
              className="w-full max-w-xs"
            >
              Continue
            </Button>
          </>
        )}
      </div>
    </FadeIn>
  );
}
