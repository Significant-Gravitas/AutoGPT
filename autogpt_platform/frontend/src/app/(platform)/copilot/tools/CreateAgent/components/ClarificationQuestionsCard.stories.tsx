import type { Meta, StoryObj } from "@storybook/nextjs";
import { ClarificationQuestionsCard } from "./ClarificationQuestionsCard";

const meta: Meta<typeof ClarificationQuestionsCard> = {
  title: "CoPilot/Tools/CreateAgent/ClarificationQuestionsCard",
  component: ClarificationQuestionsCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Interactive card that asks clarifying questions before creating an agent.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 600 }}>
        <Story />
      </div>
    ),
  ],
  args: {
    onSubmitAnswers: (answers) =>
      console.log("[Storybook] onSubmitAnswers:", answers),
  },
};
export default meta;
type Story = StoryObj<typeof ClarificationQuestionsCard>;

export const SingleQuestion: Story = {
  args: {
    message: "I need a bit more detail to build the right agent for you.",
    questions: [
      {
        question: "What data source should the agent pull from?",
        keyword: "data_source",
        example: "e.g. a REST API, a CSV file, or a database",
      },
    ],
  },
};

export const MultipleQuestions: Story = {
  args: {
    message:
      "Before I create your agent, I have a few questions to make sure it does exactly what you need.",
    questions: [
      {
        question: "What is the primary goal of this agent?",
        keyword: "goal",
        example: "e.g. monitor stock prices and send alerts",
      },
      {
        question: "How often should it run?",
        keyword: "frequency",
        example: "e.g. every hour, once a day",
      },
      {
        question: "Where should it send notifications?",
        keyword: "notification_channel",
        example: "e.g. email, Slack, Discord",
      },
    ],
  },
};

export const Answered: Story = {
  args: {
    message: "Thanks for your answers!",
    questions: [
      {
        question: "What is the primary goal?",
        keyword: "goal",
      },
    ],
    isAnswered: true,
  },
};
