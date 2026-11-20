"use client";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Text } from "@/components/atoms/Text/Text";
import { useAccessStep } from "./useAccessStep";

/**
 * Right after signing up with a provider whose data Otto can use, offers that
 * connection. Its place in the wizard comes from the step order, so it can
 * move without changing this step.
 */
export function AccessStep() {
  const { access, handleContinue } = useAccessStep();
  if (!access) return null;
  const { Connect } = access;

  return (
    <FadeIn>
      <div className="flex w-full max-w-xl flex-col items-center gap-6 px-4">
        <div className="flex flex-col items-center gap-3 text-center">
          <Text variant="h4" as="h1">
            {access.title}
          </Text>
          <Text variant="large" as="p" tone="muted" className="max-w-md">
            {access.description}
          </Text>
        </div>
        <div className="w-full">
          <Connect onDone={handleContinue} onSkip={handleContinue} />
        </div>
      </div>
    </FadeIn>
  );
}
