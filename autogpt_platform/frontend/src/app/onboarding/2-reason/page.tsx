'use client';
import OnboardingButton from "@/components/onboarding/OnboardingButton";
import { useOnboarding } from "../layout";
import OnboardingStep from "@/components/onboarding/OnboardingStep";
import { OnboardingText } from "@/components/onboarding/OnboardingText";
import OnboardingList from "@/components/onboarding/OnboardingList";

const reasons = [
  {
    label: "Content & Marketing",
    text: "Content creation, social media management, blogging, creative writing",
    id: "content_marketing"
  },
  {
    label: "Business & Workflow Automation",
    text: "Operations, task management, productivity",
    id: "business_workflow_automation"
  },
  {
    label: "Data & Research",
    text: "Data analysis, insights, research, financial operation",
    id: "data_research"
  },
  {
    label: "AI & Innovation",
    text: "AI experimentation, automation testing, advanced AI applications",
    id: "ai_innovation"
  },
  {
    label: "Personal productivity",
    text: "Automating daily tasks, organizing information, personal workflows",
    id: "personal_productivity"
  },
]

function isEmptyOrWhitespace(str: string | undefined | null): boolean {
  return !str || str.trim().length === 0;
}

export default function Page() {
  const { state, setState } = useOnboarding(2);

  return (
    <OnboardingStep backHref={"/onboarding/1-welcome"}>
      <OnboardingText className="mt-4" isHeader>What's your main reason for using AutoGPT?</OnboardingText>
      <OnboardingText className="mt-1">Select the option that best matches your needs</OnboardingText>
      <OnboardingList
        className="mt-8"
        elements={reasons}
        selectedId={state.usageReason}
        onSelect={(usageReason) => setState({ usageReason })} />
      <OnboardingButton
        className="mt-10"
        href="/onboarding/3-integrations"
        disabled={isEmptyOrWhitespace(state.usageReason)}>
        Next
      </OnboardingButton>
    </OnboardingStep>
  );
}
