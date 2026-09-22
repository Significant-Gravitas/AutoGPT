"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Text } from "@/components/atoms/Text/Text";
import { useOnboardingWizardStore } from "../../store";
import { AutopilotScene } from "./components/AutopilotScene";
import { TeamScene } from "./components/TeamScene";
import { INTRO_SLIDES, type IntroSlideId } from "./helpers";

interface Props {
  slide: IntroSlideId;
}

const SCENES: Record<IntroSlideId, () => React.JSX.Element> = {
  team: TeamScene,
  autopilot: AutopilotScene,
};

// One intro slide as a wizard step of its own, so it has a URL, a dot, a
// place in the funnel and a Back button like every other step.
export function IntroStep({ slide }: Props) {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const { title, body } = INTRO_SLIDES[slide];
  const Scene = SCENES[slide];

  return (
    <FadeIn className="flex w-full max-w-lg flex-col items-center gap-8 px-4">
      <Text variant="h4" as="h1" className="text-center leading-tight">
        {title}
      </Text>

      <div
        data-testid="intro-stage"
        className="relative h-[340px] w-full overflow-hidden"
      >
        <Scene />
      </div>

      {body && (
        <Text
          variant="large"
          as="p"
          tone="muted"
          className="max-w-md text-center"
        >
          {body}
        </Text>
      )}

      <Button
        type="button"
        size="small"
        onClick={nextStep}
        className="h-10 w-56 rounded-xl"
      >
        Next
      </Button>
    </FadeIn>
  );
}
