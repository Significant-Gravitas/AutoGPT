"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import type { ToolUIPart } from "ai";
import { domAnimation, LazyMotion, m } from "framer-motion";
import { useContext, useId } from "react";
import { useExpertMap } from "../../useExpertMap";
import { AnsweredQuestions } from "../AnsweredQuestions/AnsweredQuestions";
import { MorphingTextAnimation } from "../MorphingTextAnimation/MorphingTextAnimation";
import { AnsweredQuestionsContext } from "../QuestionDock/AnsweredQuestionsContext";
import { OnboardingChoices } from "./components/OnboardingChoices/OnboardingChoices";
import { parseExpertOnboarding, type ExpertOnboardingOutput } from "./helpers";
import { PendingOnboardingContext } from "./PendingOnboardingContext";
import { useExpertOnboardingCard } from "./useExpertOnboardingCard";

interface Props {
  part: ToolUIPart;
}

/** The intake card a freshly hired expert opens with: a one-line greeting
 *  and a few tappable questions, one per step, Typeform style. It stands
 *  outside the tool chain because a chain collapses on top of its rows, and
 *  this one is the only thing on screen the user is meant to act on. */
export function ExpertOnboardingCard({ part }: Props) {
  const pendingCallId = useContext(PendingOnboardingContext);
  const onboarding = parseExpertOnboarding(part);

  if (!onboarding) {
    // A settled call that parsed to nothing is as final as an errored one —
    // no later output is coming, so the pending line would spin forever.
    const isSettled =
      part.state === "output-error" || part.state === "output-available";
    if (isSettled) {
      return (
        <div className="py-2 text-sm text-zinc-500">
          Couldn&apos;t open the setup questions.
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 py-2 text-sm text-muted-foreground">
        <MorphingTextAnimation text="Getting set up…" />
      </div>
    );
  }

  return (
    <OnboardingForm
      callId={part.toolCallId}
      onboarding={onboarding}
      isLive={pendingCallId === part.toolCallId}
    />
  );
}

interface FormProps {
  callId: string;
  onboarding: ExpertOnboardingOutput;
  isLive: boolean;
}

function OnboardingForm({ callId, onboarding, isLive }: FormProps) {
  const { expertsById } = useExpertMap();
  const answers = useContext(AnsweredQuestionsContext).get(callId);
  const sectionId = useId();
  const {
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
  } = useExpertOnboardingCard({
    callId,
    steps: onboarding.steps,
    isLive,
  });

  const expert = onboarding.expertId
    ? expertsById.get(onboarding.expertId)
    : undefined;
  const name = expert?.name ?? "Your new hire";
  const labelId = `${sectionId}-${currentStep.keyword}`;

  if (isDone) {
    // Skipped questions have no answer to show, so only answered ones stay.
    const answered = onboarding.steps.flatMap((step) => {
      const answer = answers?.get(step.keyword);
      return answer
        ? [{ key: step.keyword, question: step.question, answer }]
        : [];
    });
    return (
      <div className="flex flex-col gap-1 py-2">
        <div className="flex items-center gap-2 text-sm text-zinc-500">
          <Icon
            icon={CheckmarkCircle02Icon}
            size={16}
            className="shrink-0 text-zinc-400"
          />
          Setup questions from {name}
        </div>
        {answered.length > 0 && <AnsweredQuestions items={answered} />}
      </div>
    );
  }

  const total = onboarding.steps.length;

  return (
    <div className="w-full max-w-xl overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)]">
      <div
        role="progressbar"
        aria-label="Setup progress"
        aria-valuemin={1}
        aria-valuemax={total}
        aria-valuenow={current + 1}
        aria-valuetext={`Question ${current + 1} of ${total}`}
        className="h-1 bg-zinc-100"
      >
        <div
          className="h-full bg-zinc-800 transition-[width] duration-300 ease-out"
          style={{ width: `${((current + 1) / total) * 100}%` }}
        />
      </div>

      {/* Wraps only the pager, not the whole card: ``strict`` rejects any
            ``motion`` component below it. */}
      <LazyMotion features={domAnimation} strict>
        <m.div
          key={currentStep.keyword}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
          className="flex flex-col gap-4 px-5 pt-5"
        >
          <span className="flex gap-2 text-lg font-medium leading-snug text-zinc-900">
            <span
              aria-hidden="true"
              className="flex shrink-0 items-center gap-0.5 text-sm text-zinc-400"
            >
              {current + 1}
              <Icon icon={ArrowRight01Icon} size={12} />
            </span>
            <span id={labelId}>{currentStep.question}</span>
          </span>
          <OnboardingChoices
            key={currentStep.keyword}
            options={currentStep.options}
            value={value}
            labelId={labelId}
            autoFocus={current > 0}
            onChoose={choose}
            onChange={setAnswer}
            onSubmit={advance}
          />
        </m.div>
      </LazyMotion>

      <div className="flex items-center justify-between px-5 pb-4 pt-4">
        <button
          type="button"
          aria-label="Previous question"
          disabled={current === 0}
          onClick={goBack}
          className="flex size-8 items-center justify-center rounded-md text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:invisible"
        >
          <Icon icon={ArrowLeft01Icon} size={16} />
        </button>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={skip}
            disabled={isSending}
            className="flex h-8 items-center rounded-md border border-zinc-200 bg-white px-3.5 text-sm font-medium text-zinc-700 transition-colors enabled:hover:border-zinc-300 enabled:hover:bg-zinc-50 disabled:opacity-50"
          >
            Skip questions
          </button>
          <button
            type="button"
            aria-label={isLast ? "Send answers" : "Next question"}
            disabled={!isAnswered || isSending}
            onClick={advance}
            className={cn(
              "flex h-8 items-center gap-1.5 rounded-md px-4 text-sm font-medium transition-all duration-200 enabled:active:scale-95",
              isAnswered && !isSending
                ? "bg-zinc-900 text-white hover:bg-zinc-800"
                : "bg-zinc-100 text-zinc-400",
            )}
          >
            {isLast ? "Send" : "Next"}
            <Icon icon={isLast ? SentIcon : ArrowRight01Icon} size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
