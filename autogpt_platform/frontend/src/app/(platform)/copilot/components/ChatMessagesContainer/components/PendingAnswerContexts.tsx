"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { getHeldOutcomes } from "../heldCallRows";
import { HeldOutcomesContext } from "../HeldOutcomesContext";
import { getPendingOnboardingCallId } from "../../ExpertOnboardingCard/helpers";
import { PendingOnboardingContext } from "../../ExpertOnboardingCard/PendingOnboardingContext";
import { getPendingQuestions } from "../../QuestionDock/helpers";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";
import { CurrentPauseContext } from "../../TurnPausedCard/CurrentPauseContext";
import { getCurrentPause } from "../../TurnPausedCard/helpers";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  children: React.ReactNode;
}

/** What the session is still waiting on the user for: the clarifying-question
 *  dock, the hire's onboarding card, a held call's row and a Paused card each
 *  decide from here whether they are live or already history, rather than
 *  from local state a reload would lose. */
export function PendingAnswerContexts({ messages, children }: Props) {
  return (
    <PendingQuestionsContext.Provider value={getPendingQuestions(messages)}>
      <PendingOnboardingContext.Provider
        value={getPendingOnboardingCallId(messages)}
      >
        <HeldOutcomesContext.Provider value={getHeldOutcomes(messages)}>
          <CurrentPauseContext.Provider value={getCurrentPause(messages)}>
            {children}
          </CurrentPauseContext.Provider>
        </HeldOutcomesContext.Provider>
      </PendingOnboardingContext.Provider>
    </PendingQuestionsContext.Provider>
  );
}
