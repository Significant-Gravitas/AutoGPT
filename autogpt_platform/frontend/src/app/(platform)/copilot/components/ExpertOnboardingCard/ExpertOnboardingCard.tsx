"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import type { ToolUIPart } from "ai";
import { m } from "framer-motion";
import { useContext, useId } from "react";
import { useExpertMap } from "../../useExpertMap";
import { ExpertAvatar } from "../ChatMessagesContainer/components/ExpertAvatar/ExpertAvatar";
import { QuestionAnswerField } from "../ChainActionCard/QuestionAnswerField";
import { MorphingTextAnimation } from "../MorphingTextAnimation/MorphingTextAnimation";
import {
  parseExpertOnboarding,
  toClarifyingQuestion,
  type ExpertOnboardingOutput,
} from "./helpers";
import { PendingOnboardingContext } from "./PendingOnboardingContext";
import { useExpertOnboardingCard } from "./useExpertOnboardingCard";

interface Props {
  part: ToolUIPart;
}

/** The intake card a freshly hired expert opens with: a greeting in its own
 *  voice and a few tappable questions, one per step. It stands outside the
 *  tool chain because a chain collapses on top of its rows, and this one is
 *  the only thing on screen the user is meant to act on. */
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
      onboarding={onboarding}
      isLive={pendingCallId === part.toolCallId}
    />
  );
}

interface FormProps {
  onboarding: ExpertOnboardingOutput;
  isLive: boolean;
}

function OnboardingForm({ onboarding, isLive }: FormProps) {
  const { expertsById } = useExpertMap();
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
    goBack,
    setAnswer,
    skip,
  } = useExpertOnboardingCard({ steps: onboarding.steps, isLive });

  const expert = onboarding.expertId
    ? expertsById.get(onboarding.expertId)
    : undefined;
  const name = expert?.name ?? "Your new hire";
  const labelId = `${sectionId}-${currentStep.keyword}`;

  if (isDone) {
    return (
      <div className="flex items-center gap-2 py-2 text-sm text-zinc-500">
        <Icon
          icon={CheckmarkCircle02Icon}
          size={16}
          className="shrink-0 text-zinc-400"
        />
        Setup questions from {name}
      </div>
    );
  }

  return (
    <div className="w-full max-w-lg overflow-hidden rounded-3xl border border-zinc-100 bg-white shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)]">
      <div className="flex items-start justify-between gap-3 border-b border-zinc-100 px-4 py-3">
        <span className="flex min-w-0 items-center gap-3">
          <ExpertAvatar name={name} avatarUrl={expert?.avatarUrl ?? null} />
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-sm font-medium text-zinc-900">
              {name}
            </span>
            {expert?.role && (
              <span className="truncate text-xs text-zinc-500">
                {expert.role}
              </span>
            )}
          </span>
        </span>
        <button
          type="button"
          onClick={skip}
          disabled={isSending}
          className="shrink-0 rounded-full px-2 py-0.5 text-xs text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:opacity-50"
        >
          Skip
        </button>
      </div>

      {onboarding.greeting && (
        <p className="border-b border-zinc-100 px-4 py-3 text-sm leading-relaxed text-zinc-700">
          {onboarding.greeting}
        </p>
      )}

      <m.div
        key={currentStep.keyword}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
        className="flex flex-col gap-1.5 px-4 py-3"
      >
        <span id={labelId} className="text-sm text-zinc-700">
          {currentStep.question}
        </span>
        <QuestionAnswerField
          // The field owns a typing toggle that must not leak between
          // questions; the key remounts it on every step.
          key={currentStep.keyword}
          question={toClarifyingQuestion(currentStep)}
          value={value}
          labelId={labelId}
          autoFocus={current > 0}
          onChange={setAnswer}
          onSubmit={advance}
        />
      </m.div>

      <div className="flex items-center justify-between px-4 pb-3 pt-1">
        <span className="flex items-center gap-2">
          <button
            type="button"
            aria-label="Previous question"
            disabled={current === 0}
            onClick={goBack}
            className="flex size-6 items-center justify-center rounded-lg text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:opacity-35"
          >
            <Icon icon={ArrowLeft01Icon} size={14} />
          </button>
          <span
            aria-hidden="true"
            className="flex items-center gap-1.5"
            data-testid="expert-onboarding-progress"
          >
            {onboarding.steps.map((step, index) => (
              <span
                key={step.keyword}
                className={
                  "rounded-full transition-all duration-300 " +
                  (index === current
                    ? "size-2.5 border-2 border-zinc-800"
                    : index < current
                      ? "size-2 bg-zinc-400"
                      : "size-2 border border-zinc-300")
                }
              />
            ))}
          </span>
          <span className="text-xs text-zinc-400">
            {current + 1} of {onboarding.steps.length}
          </span>
        </span>

        <button
          type="button"
          aria-label={isLast ? "Send answers" : "Next question"}
          disabled={!isAnswered || isSending}
          onClick={advance}
          className={
            "flex size-8 items-center justify-center rounded-full transition-all duration-200 enabled:active:scale-95 " +
            (isAnswered && !isSending
              ? "bg-zinc-800 text-white hover:bg-zinc-900"
              : "bg-zinc-100 text-zinc-400")
          }
        >
          <Icon icon={isLast ? SentIcon : ArrowRight01Icon} size={15} />
        </button>
      </div>
    </div>
  );
}
