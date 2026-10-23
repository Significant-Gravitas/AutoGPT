"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { isComposingEvent, isKey } from "@/lib/keyboard";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  isAnswered,
  type QuestionAnswer,
} from "../../tools/clarifying-questions";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import type { OnboardingChoicesHandle } from "./components/OnboardingChoices/useOnboardingChoices";
import {
  buildOnboardingAnswersMessage,
  isTextEntryTarget,
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
  const [answers, setAnswers] = useState<Record<string, QuestionAnswer>>(
    () => readOnboardingProgress(callId)?.answers ?? {},
  );
  const [isSent, setIsSent] = useState(false);
  const [isSending, setIsSending] = useState(false);
  // Set while the user changes one answer from the review list, so Next takes
  // them straight back to it instead of through every later question.
  const [isEditing, setIsEditing] = useState(false);
  // The step a tapped option is about to move to. Saved in place of ``step``
  // so a remount inside the auto-advance beat still lands on the next
  // question instead of stranding the user on the one they just answered.
  const [advancingTo, setAdvancingTo] = useState<number | null>(null);
  const advanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const choicesRef = useRef<OnboardingChoicesHandle>(null);

  // One step past the last question lists every answer before Send.
  const reviewStep = steps.length;
  const current = Math.min(step, reviewStep);
  const isReview = current === reviewStep;
  const currentStep = isReview ? null : steps[current];
  const answer = currentStep ? answers[currentStep.keyword] : undefined;
  const isReady = isReview
    ? steps.every((s) => isAnswered(answers[s.keyword]))
    : isAnswered(answer);
  const nextStep = isEditing ? reviewStep : current + 1;
  // A card the thread has moved past renders as history, whether the answers
  // went out from this tab or from another one.
  const isDone = isSent || !isLive;

  useEffect(() => {
    if (isDone) return;
    writeOnboardingProgress(callId, { step: advancingTo ?? step, answers });
  }, [callId, step, advancingTo, answers, isDone]);

  useMountEffect(() => () => {
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
  });

  function cancelAutoAdvance() {
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
    advanceTimer.current = null;
    setAdvancingTo(null);
  }

  function goTo(index: number) {
    if (index === reviewStep) setIsEditing(false);
    setStep(index);
  }

  function setAnswer(next: QuestionAnswer) {
    if (!currentStep) return;
    cancelAutoAdvance();
    setAnswers((previous) => ({ ...previous, [currentStep.keyword]: next }));
  }

  // A tapped option is a whole answer, so it moves on by itself. Nothing goes
  // out without the review step, so the last question moves on too.
  function choose(option: string) {
    setAnswer(option);
    const next = nextStep;
    setAdvancingTo(next);
    advanceTimer.current = setTimeout(() => {
      advanceTimer.current = null;
      setAdvancingTo(null);
      goTo(next);
    }, AUTO_ADVANCE_MS);
  }

  function goBack() {
    cancelAutoAdvance();
    setIsEditing(false);
    setStep(Math.max(current - 1, 0));
  }

  function edit(index: number) {
    cancelAutoAdvance();
    setIsEditing(true);
    setStep(index);
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
    if (!isReady) return;
    cancelAutoAdvance();
    if (isReview) {
      void send(buildOnboardingAnswersMessage(steps, answers));
      return;
    }
    goTo(nextStep);
  }

  function skip() {
    void send(SKIP_MESSAGE);
  }

  // Keys for the whole card, not just the option list: letters pick, Enter
  // moves on, Backspace goes back and Escape skips. A text box keeps every
  // key for its text, and a row or button that handled a key itself has
  // already called preventDefault.
  function handleKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.defaultPrevented || isComposingEvent(event)) return;
    if (isTextEntryTarget(event.target)) return;
    if (isKey(event, "Escape")) {
      event.preventDefault();
      skip();
      return;
    }
    if (isKey(event, "Backspace")) {
      event.preventDefault();
      if (current > 0) goBack();
      return;
    }
    // On a button Enter is that button's click.
    if (isKey(event, "Enter")) {
      if (event.target !== event.currentTarget) return;
      event.preventDefault();
      advance();
      return;
    }
    choicesRef.current?.pickLetter(event);
  }

  return {
    answer,
    answers,
    choicesRef,
    current,
    currentStep,
    isDone,
    isReady,
    isReview,
    isSending,
    isNextReview: !isReview && nextStep === reviewStep,
    advance,
    choose,
    edit,
    goBack,
    handleKeyDown,
    setAnswer,
    skip,
  };
}
