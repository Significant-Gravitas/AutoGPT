"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { getHeldOutcomes } from "../../ChatMessagesContainer/heldCallRows";
import { HeldOutcomesContext } from "../../ChatMessagesContainer/HeldOutcomesContext";
import { getPendingOnboardingCallId } from "../../ExpertOnboardingCard/helpers";
import { PendingOnboardingContext } from "../../ExpertOnboardingCard/PendingOnboardingContext";
import { getPendingQuestions } from "../../QuestionDock/helpers";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";
import { ChatSessionContext } from "./ChatSessionContext";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  sessionId?: string | null;
  children: React.ReactNode;
}

/** What the session is still waiting on the user for: the clarifying-question
 *  dock, the hire's onboarding card and a held call's row each decide from
 *  here whether they are live or already history, rather than from local
 *  state a reload would lose. */
export function PendingAnswerContexts({
  messages,
  sessionId = null,
  children,
}: Props) {
  return (
    <ChatSessionContext.Provider value={sessionId}>
      <PendingQuestionsContext.Provider value={getPendingQuestions(messages)}>
        <PendingOnboardingContext.Provider
          value={getPendingOnboardingCallId(messages)}
        >
          <HeldOutcomesContext.Provider value={getHeldOutcomes(messages)}>
            {children}
          </HeldOutcomesContext.Provider>
        </PendingOnboardingContext.Provider>
      </PendingQuestionsContext.Provider>
    </ChatSessionContext.Provider>
  );
}
