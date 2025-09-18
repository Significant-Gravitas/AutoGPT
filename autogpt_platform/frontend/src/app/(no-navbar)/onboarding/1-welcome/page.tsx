"use client";
import { OnboardingText } from "@/components/onboarding/OnboardingText";
import OnboardingButton from "@/components/onboarding/OnboardingButton";
import Image from "next/image";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { useEffect } from "react";
import {
  useTrackEvent,
  EventKeys,
} from "@/services/feature-flags/use-track-event";

export default function Page() {
  useOnboarding(1);
  const { track } = useTrackEvent();

  useEffect(() => {
    // Track onboarding started
    track(EventKeys.ONBOARDING_STARTED, {
      timestamp: new Date().toISOString(),
    });
  }, [track]);

  return (
    <>
      <Image
        src="/gpt_dark_RGB.svg"
        alt="GPT Dark Logo"
        className="-mb-2"
        width={300}
        height={300}
      />
      <OnboardingText className="mb-3" variant="header" center>
        Welcome to AutoGPT
      </OnboardingText>
      <OnboardingText className="mb-12" center>
        Think of AutoGPT as your digital teammate, working intelligently to
        <br />
        complete tasks based on your directions. Let&apos;s learn a bit about
        you to
        <br />
        tailor your experience.
      </OnboardingText>
      <OnboardingButton href="/onboarding/2-reason">Continue</OnboardingButton>
    </>
  );
}
