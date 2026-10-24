import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ReactNode } from "react";
import { ProgressBar } from "../../../components/ProgressBar";
import { StepIndicator } from "../../../components/StepIndicator";
import { PaidWelcome } from "./PaidWelcome";
import { TrialWelcome } from "./TrialWelcome/TrialWelcome";
import type { ReadyExpert } from "./useReadyExperts";

const OFFER: TrialOfferResponse = {
  token: "story",
  version: "1",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 5000,
  currency: "usd",
  onboarding_credit_amount: 0,
};

const EXPERTS: ReadyExpert[] = [
  {
    id: "maria",
    name: "Maria",
    avatarUrl: "/autogpt-characters/v1.1/expert-maria/neutral/256.png",
  },
  {
    id: "mina",
    name: "Mina",
    avatarUrl: "/autogpt-characters/v1.1/expert-mina/neutral/256.png",
  },
];

function OnboardingFrame({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex min-h-screen w-full flex-col items-center bg-gray-100">
      <ProgressBar currentStep={6} totalSteps={6} />
      <div className="flex w-full min-w-0 flex-1 items-center justify-center pb-8 pt-14">
        {children}
      </div>
      <div className="pb-8">
        <StepIndicator totalSteps={6} currentStep={6} />
      </div>
    </div>
  );
}

const meta = {
  title: "Onboarding/Welcome",
  parameters: { layout: "fullscreen" },
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

const noop = () => undefined;

export const Trial: Story = {
  render: () => (
    <OnboardingFrame>
      <TrialWelcome
        offer={OFFER}
        experts={EXPERTS}
        isStarting={false}
        isBlocked={false}
        error={null}
        onStart={noop}
      />
    </OnboardingFrame>
  ),
};

export const TrialOttoOnly: Story = {
  render: () => (
    <OnboardingFrame>
      <TrialWelcome
        offer={OFFER}
        experts={[]}
        isStarting={false}
        isBlocked={false}
        error={null}
        onStart={noop}
      />
    </OnboardingFrame>
  ),
};

export const Paid: Story = {
  render: () => (
    <OnboardingFrame>
      <PaidWelcome
        experts={EXPERTS}
        price="$50"
        cadence="month"
        isProcessing={false}
        isBlocked={false}
        onContinue={noop}
      />
    </OnboardingFrame>
  ),
};
