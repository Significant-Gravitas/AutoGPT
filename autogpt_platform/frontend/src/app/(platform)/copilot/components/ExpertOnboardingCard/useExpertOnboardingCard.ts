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

  function goBack() {
    setStep(Math.max(current - 1, 0));
  }

  function advance() {
    if (!isAnswered) return;
    if (!isLast) {
      setStep(current + 1);
      return;
    }
    setIsSent(true);
    void onSend(buildOnboardingAnswersMessage(steps, answers));
  }

  function skip() {
    setIsSent(true);
    void onSend(SKIP_MESSAGE);
  }

  return {
    answers,
    current,
    currentStep,
    isAnswered,
    isDone,
    isLast,
    value,
    advance,
    goBack,
    setAnswer,
    skip,
  };
}
