"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import type { QuestionAnswer } from "../../../tools/clarifying-questions";
import { getAnswerSummary, type ExpertOnboardingStep } from "../helpers";

interface Props {
  steps: ExpertOnboardingStep[];
  answers: Record<string, QuestionAnswer>;
  labelId: string;
  onEdit: (index: number) => void;
}

/** The step before Send: every answer on one screen, each a way back to its
 *  question, since sending is the one move the user can't take back. */
export function OnboardingReview({ steps, answers, labelId, onEdit }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <span
        id={labelId}
        className="text-base font-medium leading-snug text-foreground-strong sm:text-lg"
      >
        Check your answers
      </span>
      <ul aria-labelledby={labelId} className="flex flex-col gap-2">
        {steps.map((step, index) => (
          <li key={step.keyword}>
            <button
              type="button"
              onClick={() => onEdit(index)}
              // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
              className="group flex w-full items-start gap-3 rounded-md bg-surface-muted px-3 py-2.5 text-left outline-none ring-1 ring-separator transition-colors hover:bg-surface-hover focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 [@media(hover:none)]:min-h-11"
            >
              <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                <span className="sr-only">Change your answer to </span>
                <span className="text-sm text-foreground-tertiary">
                  {step.question}
                </span>
                <span className="break-words text-base text-foreground-strong sm:text-body-md">
                  {getAnswerSummary(answers[step.keyword]) || "No answer yet"}
                </span>
              </span>
              <Icon
                icon={PencilEdit02Icon}
                size={16}
                className="mt-0.5 shrink-0 text-foreground-subtle transition-colors group-hover:text-foreground-secondary"
              />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
