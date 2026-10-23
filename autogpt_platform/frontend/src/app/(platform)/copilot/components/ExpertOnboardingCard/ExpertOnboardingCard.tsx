"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import type { ToolUIPart } from "ai";
import { domAnimation, LazyMotion, m } from "framer-motion";
import { useContext, useId } from "react";
import { useExpertMap } from "../../useExpertMap";
import { AnsweredQuestions } from "../AnsweredQuestions/AnsweredQuestions";
import { MorphingTextAnimation } from "../MorphingTextAnimation/MorphingTextAnimation";
import { AnsweredQuestionsContext } from "../QuestionDock/AnsweredQuestionsContext";
import { OnboardingChoices } from "./components/OnboardingChoices/OnboardingChoices";
import { OnboardingFooter } from "./components/OnboardingFooter";
import { OnboardingReview } from "./components/OnboardingReview";
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
  // What the user's reply in the chat said, as opposed to the form's answers.
  const sentAnswers = useContext(AnsweredQuestionsContext).get(callId);
  const sectionId = useId();
  const {
    answer,
    answers,
    choicesRef,
    current,
    currentStep,
    isDone,
    isNextReview,
    isReady,
    isReview,
    isSending,
    advance,
    choose,
    edit,
    goBack,
    handleKeyDown,
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
  const labelId = `${sectionId}-${currentStep?.keyword ?? "review"}`;

  if (isDone) {
    // Skipped questions have no answer to show, so only answered ones stay.
    const answered = onboarding.steps.flatMap((step) => {
      const answer = sentAnswers?.get(step.keyword);
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
  const shown = Math.min(current + 1, total);

  return (
    // Focusable so a click anywhere on the card brings its keys along.
    <div
      tabIndex={-1}
      onKeyDown={handleKeyDown}
      className="w-full max-w-xl overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)] outline-none"
    >
      <div
        role="progressbar"
        aria-label="Setup progress"
        aria-valuemin={1}
        aria-valuemax={total}
        aria-valuenow={shown}
        aria-valuetext={
          isReview ? "Check your answers" : `Question ${shown} of ${total}`
        }
        className="h-1 bg-zinc-100"
      >
        <div
          className="h-full bg-zinc-800 transition-[width] duration-300 ease-out"
          style={{ width: `${(shown / total) * 100}%` }}
        />
      </div>

      {/* Wraps only the pager, not the whole card: ``strict`` rejects any
            ``motion`` component below it. */}
      <LazyMotion features={domAnimation} strict>
        <m.div
          key={currentStep?.keyword ?? "review"}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
          className="flex flex-col gap-4 px-4 pt-4 sm:px-5 sm:pt-5"
        >
          {currentStep ? (
            <>
              <span className="flex flex-col gap-1">
                <span className="flex gap-2 text-base font-medium leading-snug text-zinc-900 sm:text-lg">
                  <span
                    aria-hidden="true"
                    className="flex shrink-0 items-center gap-0.5 text-sm text-zinc-400"
                  >
                    {current + 1}
                    <Icon icon={ArrowRight01Icon} size={12} />
                  </span>
                  <span id={labelId}>{currentStep.question}</span>
                </span>
                {currentStep.allow_multiple && (
                  <span className="text-sm text-zinc-500">
                    Choose as many as you like
                  </span>
                )}
              </span>
              <OnboardingChoices
                key={currentStep.keyword}
                ref={choicesRef}
                options={currentStep.options}
                answer={answer}
                isMultiple={currentStep.allow_multiple}
                labelId={labelId}
                onChoose={choose}
                onChange={setAnswer}
                onSubmit={advance}
              />
            </>
          ) : (
            <OnboardingReview
              steps={onboarding.steps}
              answers={answers}
              labelId={labelId}
              onEdit={edit}
            />
          )}
        </m.div>
      </LazyMotion>

      <OnboardingFooter
        canGoBack={current > 0}
        isReady={isReady}
        isReview={isReview}
        isNextReview={isNextReview}
        isSending={isSending}
        onBack={goBack}
        onSkip={skip}
        onNext={advance}
      />
    </div>
  );
}
