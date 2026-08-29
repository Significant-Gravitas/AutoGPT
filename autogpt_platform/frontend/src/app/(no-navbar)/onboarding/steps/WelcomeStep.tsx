"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { useOnboardingWizardStore } from "../store";

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
    <FadeIn>
      <form
        onSubmit={handleSubmit}
        className="flex w-full max-w-lg flex-col items-center gap-4 px-4 md:gap-8"
      >
        <div className="mb-8 flex flex-col items-center gap-3 text-center md:mb-0">
          <AutoGPTLogo
            className="relative right-[3rem] h-24 w-[12rem]"
            hideText
          />
          <Text variant="h3">Welcome to AutoGPT</Text>
          <Text variant="lead" as="span" className="!text-zinc-500">
            Let&apos;s personalize your experience so{" "}
            <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
              AutoPilot
            </span>{" "}
            can start saving you time
          </Text>
        </div>

        <Input
          id="first-name"
          label="What should I call you?"
          placeholder="e.g. John"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full"
          autoFocus
        />

        <Button
          type="submit"
          disabled={!name.trim()}
          className="w-full max-w-xs"
        >
          Continue
        </Button>
      </form>
    </FadeIn>
  );
}
