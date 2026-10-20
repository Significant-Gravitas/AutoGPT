import type { Meta, StoryObj } from "@storybook/nextjs";
import { domAnimation, LazyMotion } from "framer-motion";
import { useState } from "react";
import {
  type ClarifyingQuestion,
  isAnswered,
  type QuestionAnswer,
} from "../../tools/clarifying-questions";
import { QuestionsSection } from "./QuestionsSection";

interface CardProps {
  questions: ClarifyingQuestion[];
  answers?: Record<string, QuestionAnswer>;
}

/** The section as the chain renders it: inside the action card's chrome and the
 *  tool chain's LazyMotion, with the answer state the chain normally owns so
 *  picks actually stick and readiness derived from it the way the card does. */
function QuestionsCard({ questions, answers: initial = {} }: CardProps) {
  const [answers, setAnswers] =
    useState<Record<string, QuestionAnswer>>(initial);
  const isReady =
    questions.length > 0 &&
    questions.every((q) => isAnswered(answers[q.keyword]));
  return (
    <LazyMotion features={domAnimation} strict>
      <div className="w-[34rem] overflow-hidden rounded-3xl border border-zinc-100 bg-white">
        <QuestionsSection
          requests={[
            {
              id: "questions-1",
              questions,
              answers,
              onAnswer: (keyword, value) =>
                setAnswers((prev) => ({ ...prev, [keyword]: value })),
              onSkip: () => undefined,
            },
          ]}
          isReady={isReady}
          onProceed={() => undefined}
        />
      </div>
    </LazyMotion>
  );
}

const meta: Meta<typeof QuestionsCard> = {
  title: "Copilot/QuestionsSection",
  component: QuestionsCard,
  parameters: {
    docs: {
      description: {
        component:
          'The copilot\'s "Answer a few questions" card. One question per step: single-select options are radios that advance the pager on a pick, multi-select ones are checkboxes that stay put so several can be ticked, and either can be answered in free text instead.',
      },
    },
  },
};
export default meta;

type Story = StoryObj<typeof QuestionsCard>;

export const SingleSelect: Story = {
  args: {
    questions: [
      {
        question: "Which channel should it post to?",
        keyword: "channel",
        options: ["Email", "Slack", "Google Docs"],
      },
    ],
  },
};

export const MultiSelect: Story = {
  args: {
    questions: [
      {
        question: "What areas should they own? Pick roles",
        keyword: "areas",
        options: ["Research", "Outreach", "Reporting", "Scheduling"],
        allow_multiple: true,
      },
    ],
  },
};

export const MultiSelectAnswered: Story = {
  args: {
    ...MultiSelect.args,
    answers: { areas: { selected: ["Research", "Reporting"], custom: "" } },
  },
};
