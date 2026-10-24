"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import isEqual from "lodash/isEqual";
import { useState } from "react";
import { getHeldOutcomes, type HeldOutcome } from "../heldCallRows";
import { HeldOutcomesContext } from "../HeldOutcomesContext";
import { getPendingOnboardingCallId } from "../../ExpertOnboardingCard/helpers";
import { PendingOnboardingContext } from "../../ExpertOnboardingCard/PendingOnboardingContext";
import { AnsweredQuestionsContext } from "../../QuestionDock/AnsweredQuestionsContext";
import { getAnsweredQuestions } from "../../QuestionDock/answeredQuestions";
import { getPendingQuestions } from "../../QuestionDock/helpers";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";
import { CurrentPauseContext } from "../../TurnPausedCard/CurrentPauseContext";
import { getCurrentPause } from "../../TurnPausedCard/helpers";
import {
  AnsweredCardsContext,
  type AnsweredCardState,
} from "../../ToolChain/AnsweredCardsContext";
import { getAnsweredCards } from "../../../cardDecisions";
import type { TurnStatsMap } from "../../../helpers/convertChatSessionToUiMessages";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  children: React.ReactNode;
  readOnly?: boolean;
  /** When each row was saved, for the time an answered card shows. */
  turnStats?: TurnStatsMap;
  /** Messages waiting behind a running turn; a decision among them stands. */
  queuedMessages?: string[];
}

const NO_HELD_OUTCOMES: ReadonlyMap<string, HeldOutcome> = new Map();

/** What the session is still waiting on the user for: the clarifying-question
 *  dock, the hire's onboarding card, a held call's row, a Paused card and a
 *  proposal card each decide from here whether they are live or already
 *  history, rather than from local state a reload would lose. A question card
 *  that is history reads the user's answers from here too. */
export function PendingAnswerContexts({
  messages,
  children,
  readOnly = false,
  turnStats,
  queuedMessages,
}: Props) {
  // The questions and held outcomes are rebuilt as new objects on every call,
  // and `messages` changes on every streamed flush, so each one keeps the
  // object it last published while its content is the same; otherwise every
  // consumer re-renders on every flush. The onboarding call id is a string
  // and the pause is the part's own `data` object (cards compare it by
  // reference), so both are passed through as they are.
  const pendingQuestions = useEqualValue(getPendingQuestions(messages));
  const answeredQuestions = useEqualValue(getAnsweredQuestions(messages));
  const heldOutcomes = useEqualValue(getHeldOutcomesOrNone(messages));
  const answeredCards = useEqualValue(
    toAnsweredCardStates(messages, turnStats, queuedMessages),
  );
  if (readOnly) {
    return (
      <AnsweredQuestionsContext.Provider value={answeredQuestions}>
        <AnsweredCardsContext.Provider value={answeredCards}>
          {children}
        </AnsweredCardsContext.Provider>
      </AnsweredQuestionsContext.Provider>
    );
  }
  return (
    <PendingQuestionsContext.Provider value={pendingQuestions}>
      <AnsweredQuestionsContext.Provider value={answeredQuestions}>
        <PendingOnboardingContext.Provider
          value={getPendingOnboardingCallId(messages)}
        >
          <HeldOutcomesContext.Provider value={heldOutcomes}>
            <CurrentPauseContext.Provider value={getCurrentPause(messages)}>
              <AnsweredCardsContext.Provider value={answeredCards}>
                {children}
              </AnsweredCardsContext.Provider>
            </CurrentPauseContext.Provider>
          </HeldOutcomesContext.Provider>
        </PendingOnboardingContext.Provider>
      </AnsweredQuestionsContext.Provider>
    </PendingQuestionsContext.Provider>
  );
}

function getHeldOutcomesOrNone(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  const outcomes = getHeldOutcomes(messages);
  return outcomes.size === 0 ? NO_HELD_OUTCOMES : outcomes;
}

function toAnsweredCardStates(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  turnStats: TurnStatsMap | undefined,
  queuedMessages: string[] | undefined,
): ReadonlyMap<string, AnsweredCardState> {
  const states = new Map<string, AnsweredCardState>();
  for (const [id, card] of getAnsweredCards(messages, queuedMessages)) {
    states.set(id, {
      verdict: card.verdict,
      at: turnStats?.get(card.messageID)?.createdAt ?? null,
      messageID: card.messageID,
    });
  }
  return states;
}

// Kept in state, not a ref: a render React throws away cannot change it.
// Setting state while rendering re-runs this component before it commits.
function useEqualValue<T>(value: T): T {
  const [kept, setKept] = useState(() => value);
  if (kept === value || isEqual(kept, value)) return kept;
  setKept(() => value);
  return value;
}
