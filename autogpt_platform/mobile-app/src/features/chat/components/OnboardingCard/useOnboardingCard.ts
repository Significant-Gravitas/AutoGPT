import { useState } from "react";

import {
  buildAnswersMessage,
  isAnswered,
  pickOption,
  setCustomText,
  SKIP_MESSAGE,
  type OnboardingStep,
  type StepAnswer,
} from "./helpers";

interface Args {
  steps: OnboardingStep[];
  onSend: (text: string) => boolean;
}

export function useOnboardingCard({ steps, onSend }: Args) {
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<Record<string, StepAnswer>>({});
  // The message that went out, so the card settles before the reply row lands.
  const [sentText, setSentText] = useState<string | null>(null);

  const step = steps[current];
  const answer = answers[step.keyword];
  const isLast = current === steps.length - 1;
  const answeredCount = steps.filter((s) => isAnswered(answers[s.keyword])).length;

  function update(next: StepAnswer) {
    setAnswers((previous) => ({ ...previous, [step.keyword]: next }));
  }

  function choose(option: string) {
    update(pickOption(step, answer, option));
  }

  function changeText(text: string) {
    update(setCustomText(step, answer, text));
  }

  function send(text: string) {
    if (onSend(text)) setSentText(text);
  }

  function next() {
    if (!isAnswered(answer)) return;
    if (isLast) {
      send(buildAnswersMessage(steps, answers));
      return;
    }
    setCurrent(current + 1);
  }

  function back() {
    setCurrent(Math.max(current - 1, 0));
  }

  function skip() {
    send(SKIP_MESSAGE);
  }

  return {
    step,
    answer,
    current,
    answeredCount,
    isLast,
    isReady: isAnswered(answer),
    sentText,
    choose,
    changeText,
    next,
    back,
    skip,
  };
}
