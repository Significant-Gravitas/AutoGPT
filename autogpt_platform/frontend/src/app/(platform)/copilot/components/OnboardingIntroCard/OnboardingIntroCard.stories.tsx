import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ComponentProps } from "react";
import { OnboardingIntroCard } from "./OnboardingIntroCard";

type Suggestions = ComponentProps<typeof OnboardingIntroCard>["suggestions"];

const TASKS: Suggestions["tasks"] = [
  {
    id: "task-1",
    title: "Summarise my inbox every morning",
    prompt: "Summarise my inbox every morning at 8am.",
    icon: "calendar",
    why: "Because mornings go on email.",
    expert_template_id: "tpl-maria",
    expert_id: "expert-maria",
    state: "open",
  },
  {
    id: "task-2",
    title: "Draft a weekly update for the team",
    prompt: "Draft a weekly update for the team from this week's notes.",
    icon: "file",
    state: "open",
  },
];

const SUGGESTIONS: Suggestions = {
  isStored: true,
  allTasks: TASKS,
  tasks: TASKS,
  experts: [
    {
      id: "expert-maria",
      name: "Maria",
      avatarUrl: null,
      role: "Marketing",
      isArchived: false,
      readOnlyReason: null,
    },
  ],
  expertsById: new Map(),
  isRosterLoaded: true,
  sendingId: null,
  sendTask: () => undefined,
  removeTask: () => undefined,
};

const COLLAPSE: ComponentProps<typeof OnboardingIntroCard>["collapse"] = {
  isCollapsed: false,
  toggle: () => undefined,
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
    collapse: COLLAPSE,
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
