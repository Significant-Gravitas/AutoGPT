"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { cn } from "@/lib/utils";
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
    <FadeIn
      className={cn("w-full", slide === "team" ? "max-w-2xl" : "max-w-lg")}
    >
      <div className="flex w-full flex-col items-center gap-8 px-4">
        <h1 className="text-center font-poppins text-heading-sm font-medium leading-tight text-foreground-base">
          {title}
        </h1>

        <div
          data-testid="intro-stage"
          className={cn(
            "relative w-full overflow-hidden",
            slide === "team" ? "h-[420px]" : "h-[340px]",
          )}
        >
          <Scene />
        </div>

        {body && (
          <p className="max-w-md text-center font-sans text-body-lg font-normal text-foreground-tertiary md:max-w-full md:text-lg md:leading-tight">
            {body}
          </p>
        )}

        <Button
          type="button"
          size="small"
          onClick={nextStep}
          className="h-10 w-56"
        >
          Next
        </Button>
      </div>
    </FadeIn>
  );
}
