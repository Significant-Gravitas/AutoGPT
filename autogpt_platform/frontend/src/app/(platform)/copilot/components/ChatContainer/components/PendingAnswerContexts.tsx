"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { getPendingOnboardingCallId } from "../../ExpertOnboardingCard/helpers";
import { PendingOnboardingContext } from "../../ExpertOnboardingCard/PendingOnboardingContext";
import { getPendingQuestions } from "../../QuestionDock/helpers";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  children: React.ReactNode;
}

/** What the session is still waiting on the user for: the clarifying-question
 *  dock and the hire's onboarding card each decide from here whether their
 *  form is live or already history, rather than from local state a reload
 *  would lose. */
export function PendingAnswerContexts({ messages, children }: Props) {
  return (
    <PendingQuestionsContext.Provider value={getPendingQuestions(messages)}>
      <PendingOnboardingContext.Provider
        value={getPendingOnboardingCallId(messages)}
      >
        {children}
      </PendingOnboardingContext.Provider>
    </PendingQuestionsContext.Provider>
  );
}
