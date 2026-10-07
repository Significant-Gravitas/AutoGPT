"use client";

import { useEffect, useRef, useState } from "react";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  buildOnboardingAnswersMessage,
  type ExpertOnboardingStep,
} from "./helpers";
import {
  clearOnboardingProgress,
  readOnboardingProgress,
  writeOnboardingProgress,
} from "./onboardingProgress";

const SKIP_MESSAGE = "Let's skip the setup questions for now.";
// Long enough to see the tapped row light up before the next question.
const AUTO_ADVANCE_MS = 300;

interface Args {
  callId: string;
  steps: ExpertOnboardingStep[];
  isLive: boolean;
}

export function useExpertOnboardingCard({ callId, steps, isLive }: Args) {
  const { onSend } = useCopilotChatActions();
  // Seeded from sessionStorage: the card is remounted when the kickoff turn
  // settles and its row is re-keyed, and again on a reload. Either would
  // otherwise throw away the answers given so far.
  const [step, setStep] = useState(
    () => readOnboardingProgress(callId)?.step ?? 0,
  );
  const [answers, setAnswers] = useState<Record<string, string>>(
    () => readOnboardingProgress(callId)?.answers ?? {},
  );
  const [isSent, setIsSent] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const advanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const current = Math.min(step, steps.length - 1);
  const currentStep = steps[current];
  const value = answers[currentStep.keyword] ?? "";
  const isAnswered = value.trim().length > 0;
  const isLast = current === steps.length - 1;
  // A card the thread has moved past renders as history, whether the answers
  // went out from this tab or from another one.
  const isDone = isSent || !isLive;

  useEffect(() => {
    if (isDone) return;
    writeOnboardingProgress(callId, { step, answers });
  }, [callId, step, answers, isDone]);

  function cancelAutoAdvance() {
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
    advanceTimer.current = null;
  }

  function setAnswer(next: string) {
    cancelAutoAdvance();
    setAnswers((previous) => ({ ...previous, [currentStep.keyword]: next }));
  }

  // A tapped option is a whole answer, so it moves on by itself. The last
  // step waits for Send: sending is the one move the user can't take back.
  function choose(option: string) {
    setAnswer(option);
    if (isLast) return;
    advanceTimer.current = setTimeout(
      () => setStep(current + 1),
      AUTO_ADVANCE_MS,
    );
  }

  function goBack() {
    cancelAutoAdvance();
    setStep(Math.max(current - 1, 0));
  }

  // Settling the card is what removes the user's only copy of their answers,
  // so it waits for the send to resolve. A rejected send (no session, dispatch
  // failure) leaves the form exactly as it was, still submittable.
  async function send(message: string) {
    if (isSending) return;
    setIsSending(true);
    try {
      await onSend(message);
      clearOnboardingProgress(callId);
      setIsSent(true);
    } catch {
      setIsSent(false);
    } finally {
      setIsSending(false);
    }
  }

  function advance() {
    if (!isAnswered) return;
    if (!isLast) {
      setStep(current + 1);
      return;
    }
    void send(buildOnboardingAnswersMessage(steps, answers));
  }

  function skip() {
    void send(SKIP_MESSAGE);
  }

  return {
    answers,
    current,
    currentStep,
    isAnswered,
    isDone,
    isLast,
    isSending,
    value,
    advance,
    choose,
    goBack,
    setAnswer,
    skip,
  };
}
