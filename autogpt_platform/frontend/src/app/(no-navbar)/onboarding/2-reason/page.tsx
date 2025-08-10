"use client";
import OnboardingButton from "@/components/onboarding/OnboardingButton";
import {
  OnboardingFooter,
  OnboardingHeader,
  OnboardingStep,
} from "@/components/onboarding/OnboardingStep";
import { OnboardingText } from "@/components/onboarding/OnboardingText";
import OnboardingList from "@/components/onboarding/OnboardingList";
import { isEmptyOrWhitespace } from "@/lib/utils";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";

const reasons = [
  {
    label: "Content & Marketing",
    text: "Content creation, social media management, blogging, creative writing",
    id: "content_marketing",
  },
  {
    label: "Business & Workflow Automation",
    text: "Operations, task management, productivity",
    id: "business_workflow_automation",
  },
  {
    label: "Data & Research",
    text: "Data analysis, insights, research, financial operation",
    id: "data_research",
  },
  {
    label: "AI & Innovation",
    text: "AI experimentation, automation testing, advanced AI applications",
    id: "ai_innovation",
  },
  {
    label: "Personal productivity",
    text: "Automating daily tasks, organizing information, personal workflows",
    id: "personal_productivity",
  },
];

export default function Page() {
  const { state, updateState } = useOnboarding(2, "WELCOME");

  return (
    <OnboardingStep>
      <OnboardingHeader backHref={"/onboarding/1-welcome"}>
        <OnboardingText className="mt-4" variant="header" center>
          What&apos;s your main reason for using AutoGPT?
        </OnboardingText>
        <OnboardingText className="mt-1" center>
          Select the option that best matches your needs
        </OnboardingText>
      </OnboardingHeader>
      <OnboardingList
        elements={reasons}
        selectedId={state?.usageReason}
        onSelect={(usageReason) => updateState({ usageReason })}
      />
      <OnboardingFooter>
        <OnboardingButton
          href="/onboarding/3-services"
          disabled={isEmptyOrWhitespace(state?.usageReason)}
        >
          Next
        </OnboardingButton>
      </OnboardingFooter>
    </OnboardingStep>
  );
}
