import { Button } from "@/components/atoms/Button/Button";
import { Logout03Icon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ReactNode } from "react";
import { ProgressBar } from "../../components/ProgressBar";
import { StepIndicator } from "../../components/StepIndicator";
import OnboardingLayout from "../../layout";
import { PAYWALL_LAST_STEPS } from "../../store";
import { IntroStep } from "./IntroStep";

// The hosted wizard's step count, as useOnboardingLayout derives it.
const TOTAL_STEPS = PAYWALL_LAST_STEPS.preparing - 1;

// The onboarding page's own chrome around a step (layout.tsx and page.tsx),
// fixed over Storybook's padded canvas so the slide sits where it does in
// the app.
function OnboardingFrame({
  step,
  children,
}: {
  step: number;
  children: ReactNode;
}) {
  return (
    <div className="fixed inset-0 overflow-auto">
      <OnboardingLayout>
        <div className="flex min-h-screen w-full flex-col items-center">
          <ProgressBar currentStep={step} totalSteps={TOTAL_STEPS} />
          <div className="flex w-full min-w-0 flex-1 items-center justify-center pt-16 pb-8">
            {children}
          </div>
          <div className="pb-8">
            <StepIndicator totalSteps={TOTAL_STEPS} currentStep={step} />
          </div>
          <Button
            as="NextLink"
            href="/logout"
            variant="toggle"
            size="xs"
            leadingIcon={Logout03Icon}
            className="absolute bottom-6 left-6"
          >
            Log out
          </Button>
        </div>
      </OnboardingLayout>
    </div>
  );
}

const meta = {
  title: "Onboarding/Intro",
  parameters: { layout: "fullscreen" },
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

export const Team: Story = {
  render: () => (
    <OnboardingFrame step={1}>
      <IntroStep slide="team" />
    </OnboardingFrame>
  ),
};

export const Otto: Story = {
  render: () => (
    <OnboardingFrame step={2}>
      <IntroStep slide="autopilot" />
    </OnboardingFrame>
  ),
};
