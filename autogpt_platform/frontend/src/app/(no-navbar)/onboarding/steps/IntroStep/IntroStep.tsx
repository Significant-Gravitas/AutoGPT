"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { MotionConfig } from "framer-motion";
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
  const { eyebrow, title, body, cta } = INTRO_SLIDES[slide];
  const Scene = SCENES[slide];
  // The team slide is the first screen of onboarding, so it opens with a
  // welcome and a larger title than the steps after it.
  const isWelcome = slide === "team";

  // On the welcome slide, someone who asks for reduced motion gets the
  // fade-in without its slide up.
  return (
    <MotionConfig reducedMotion={isWelcome ? "user" : "never"}>
      <FadeIn className={cn("w-full", isWelcome ? "max-w-2xl" : "max-w-lg")}>
        <div
          className={cn(
            "flex w-full flex-col items-center px-4",
            isWelcome ? "gap-0" : "gap-8",
          )}
        >
          {eyebrow && (
            <span className="mb-3 rounded-full bg-accent-surface px-3 py-1 ring-1 ring-accent-strong/15 md:mb-4">
              <Text as="span" variant="body-medium" tone="accent">
                {eyebrow}
              </Text>
            </span>
          )}

          {isWelcome ? (
            <Text
              variant="h2"
              as="h1"
              weight="semibold"
              align="center"
              className="max-w-xl text-balance"
            >
              {title}
            </Text>
          ) : (
            <h1 className="text-center font-poppins text-heading-sm leading-tight font-medium text-foreground-base">
              {title}
            </h1>
          )}

          <div
            data-testid="intro-stage"
            className={cn(
              "relative w-full",
              isWelcome
                ? "mt-6 mb-6 md:mt-10 md:mb-8"
                : "h-[340px] overflow-hidden",
            )}
          >
            <Scene />
          </div>

          {body && (
            <Text
              variant="large"
              tone="secondary"
              align="center"
              className="max-w-lg text-balance"
            >
              {body}
            </Text>
          )}

          <Button
            type="button"
            size={isWelcome ? "large" : "small"}
            onClick={nextStep}
            rightIcon={
              isWelcome ? (
                <Icon icon={ArrowRight02Icon} size={18} aria-hidden />
              ) : undefined
            }
            className={isWelcome ? "mt-6 w-60 md:mt-8" : "h-10 w-56"}
          >
            {cta}
          </Button>
        </div>
      </FadeIn>
    </MotionConfig>
  );
}
