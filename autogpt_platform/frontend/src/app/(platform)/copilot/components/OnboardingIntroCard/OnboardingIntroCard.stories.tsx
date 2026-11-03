import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ComponentProps } from "react";
import { OnboardingIntroCard } from "./OnboardingIntroCard";

type Suggestions = ComponentProps<typeof OnboardingIntroCard>["suggestions"];

const SUGGESTIONS: Suggestions = {
  isStored: true,
  tasks: [
    {
      id: "task-1",
      title: "Summarise my inbox every morning",
      prompt: "Summarise my inbox every morning at 8am.",
      icon: "calendar",
      state: "open",
    },
    {
      id: "task-2",
      title: "Draft a weekly update for the team",
      prompt: "Draft a weekly update for the team from this week's notes.",
      icon: "file",
      state: "open",
    },
  ],
  sendingId: null,
  sendTask: async () => undefined,
  hideTask: () => undefined,
  hideAll: () => undefined,
};

const meta: Meta<typeof OnboardingIntroCard> = {
  title: "Copilot/OnboardingIntroCard",
  component: OnboardingIntroCard,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "The greeting a user sees after onboarding. With a recording transcript, a copy button with a tooltip sits beside the heading.",
      },
    },
  },
  args: {
    name: "Sam",
    greeting:
      "Thanks for telling me about your week. I can take the inbox and the weekly update off your plate. Pick one below to start.",
    suggestions: SUGGESTIONS,
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="pt-12">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const WithTranscript: Story = {
  name: "With transcript (copy button)",
  args: {
    transcript:
      "I spend every Monday morning sorting my inbox and writing the team update.",
  },
};
