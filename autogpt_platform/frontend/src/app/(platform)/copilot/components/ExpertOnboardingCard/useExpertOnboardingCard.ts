"use client";

import { useState } from "react";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  buildOnboardingAnswersMessage,
  type ExpertOnboardingStep,
} from "./helpers";

const SKIP_MESSAGE = "Let's skip the setup questions for now.";

interface Args {
  steps: ExpertOnboardingStep[];
  isLive: boolean;
}

export function useExpertOnboardingCard({ steps, isLive }: Args) {
  const { onSend } = useCopilotChatActions();
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isSent, setIsSent] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const current = Math.min(step, steps.length - 1);
  const currentStep = steps[current];
  const value = answers[currentStep.keyword] ?? "";
  const isAnswered = value.trim().length > 0;
  const isLast = current === steps.length - 1;
  // A card the thread has moved past renders as history, whether the answers
  // went out from this tab or from another one.
  const isDone = isSent || !isLive;

  function setAnswer(next: string) {
    setAnswers((previous) => ({ ...previous, [currentStep.keyword]: next }));
  }

  // A tap on an option is the whole answer, so the pager moves on by itself;
  // the last question keeps the send button as its explicit final step.
  function pickAnswer(next: string) {
    setAnswer(next);
    if (!isLast) setStep(current + 1);
  }

  function goBack() {
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
    goBack,
    pickAnswer,
    setAnswer,
    skip,
  };
}
