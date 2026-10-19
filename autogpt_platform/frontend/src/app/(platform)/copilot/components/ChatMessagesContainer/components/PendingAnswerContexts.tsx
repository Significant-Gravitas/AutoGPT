"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import isEqual from "lodash/isEqual";
import { useMemo, useRef } from "react";
import { getHeldOutcomes, type HeldOutcome } from "../heldCallRows";
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

const NO_HELD_OUTCOMES: ReadonlyMap<string, HeldOutcome> = new Map();

/** What the session is still waiting on the user for: the clarifying-question
 *  dock, the hire's onboarding card, a held call's row and a Paused card each
 *  decide from here whether they are live or already history, rather than
 *  from local state a reload would lose. */
export function PendingAnswerContexts({ messages, children }: Props) {
  // The questions and held outcomes are rebuilt as new objects on every call,
  // and `messages` changes on every streamed flush, so each one keeps the
  // object it last published while its content is the same; otherwise every
  // consumer re-renders on every flush. The onboarding call id is a string
  // and the pause is the part's own `data` object (cards compare it by
  // reference), so both are passed through as they are.
  const pendingQuestions = useEqualValue(
    useMemo(() => getPendingQuestions(messages), [messages]),
  );
  const heldOutcomes = useEqualValue(
    useMemo(() => {
      const outcomes = getHeldOutcomes(messages);
      return outcomes.size === 0 ? NO_HELD_OUTCOMES : outcomes;
    }, [messages]),
  );
  return (
    <PendingQuestionsContext.Provider value={pendingQuestions}>
      <PendingOnboardingContext.Provider
        value={getPendingOnboardingCallId(messages)}
      >
        <HeldOutcomesContext.Provider value={heldOutcomes}>
          <CurrentPauseContext.Provider value={getCurrentPause(messages)}>
            {children}
          </CurrentPauseContext.Provider>
        </HeldOutcomesContext.Provider>
      </PendingOnboardingContext.Provider>
    </PendingQuestionsContext.Provider>
  );
}

function useEqualValue<T>(value: T): T {
  const kept = useRef(value);
  if (!isEqual(kept.current, value)) kept.current = value;
  return kept.current;
}
