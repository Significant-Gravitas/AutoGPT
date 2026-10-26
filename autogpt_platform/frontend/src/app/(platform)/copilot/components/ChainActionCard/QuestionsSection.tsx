"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Message01Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import { m } from "framer-motion";
import { useId, useState } from "react";
import {
  isAnswered,
  toAnswerText,
  toMultiAnswer,
} from "../../tools/clarifying-questions";
import type { QuestionRequest } from "./helpers";
import { QuestionAnswerField } from "./QuestionAnswerField";
import { QuestionMultiAnswerField } from "./QuestionMultiAnswerField";

interface Props {
  requests: QuestionRequest[];
  isReady: boolean;
  onProceed: () => void;
}

/** One question per step. Picking an option only selects it, so the user can
 *  change their mind or edit it first; the footer pager (chevrons + ring dots)
 *  moves between questions, and the round action button advances and, on the
 *  last step, sends every answer as one message. */
export function QuestionsSection({ requests, isReady, onProceed }: Props) {
  const [step, setStep] = useState(0);
  const sectionId = useId();
  // Keyed by position, not keyword: keywords are unique only within a request,
  // and a duplicate would collapse two questions onto one id — the field would
  // never remount and the second question would show the first one's answer.
  const questions = requests
    .flatMap((request) =>
      request.questions.map((question) => ({ request, question })),
    )
    .map((entry, index) => ({ ...entry, id: `${entry.request.id}-${index}` }));
  if (questions.length === 0) return null;

  const current = Math.min(step, questions.length - 1);
  const { request, question, id } = questions[current];
  const labelId = `${sectionId}-${id}`;
  const answer = request.answers[question.keyword];
  const multiOptions = question.allow_multiple ? (question.options ?? []) : [];
  const answered = isAnswered(answer);
  const isLast = current === questions.length - 1;
  const actionEnabled = isLast ? isReady : answered;

  function handleAction() {
    if (isLast) {
      if (isReady) onProceed();
    } else if (answered) {
      setStep(current + 1);
    }
  }

  return (
    <>
      <div className="flex items-center justify-between gap-2.5 border-b border-separator px-4 py-3">
        <span className="flex items-center gap-2.5">
          <Icon
            icon={Message01Icon}
            size={18}
            className="text-foreground-subtle"
          />
          <span className="text-sm font-medium text-foreground-strong">
            Answer a few questions
          </span>
        </span>
        <button
          type="button"
          onClick={() => requests.forEach((req) => req.onSkip())}
          className="rounded-full px-2 py-0.5 text-xs text-foreground-subtle transition-colors hover:bg-surface-hover hover:text-foreground-secondary"
        >
          Skip
        </button>
      </div>

      <m.div
        key={id}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
        className="flex flex-col gap-4 px-5 pb-4 pt-5"
      >
        <span
          id={labelId}
          className="text-lg font-medium leading-snug text-foreground-strong"
        >
          {question.question}
        </span>
        {multiOptions.length > 0 ? (
          <QuestionMultiAnswerField
            key={id}
            options={multiOptions}
            value={toMultiAnswer(answer)}
            labelId={labelId}
            autoFocus={current > 0}
            onChange={(value) => request.onAnswer(question.keyword, value)}
            onSubmit={handleAction}
          />
        ) : (
          <QuestionAnswerField
            // The field's typing toggle must reset per question; keying it
            // here rather than only on the wrapper keeps that contract local
            // to the component that owns the state.
            key={id}
            question={question}
            value={toAnswerText(answer)}
            labelId={labelId}
            autoFocus={current > 0}
            onChange={(value) => request.onAnswer(question.keyword, value)}
            onSubmit={handleAction}
          />
        )}
      </m.div>

      <div className="flex items-center justify-between px-5 pb-4 pt-1">
        <span className="flex items-center gap-2">
          <button
            type="button"
            aria-label="Previous question"
            disabled={current === 0}
            onClick={() => setStep(current - 1)}
            className="flex size-6 items-center justify-center rounded-lg text-foreground-subtle transition-colors enabled:hover:bg-surface-hover enabled:hover:text-foreground-secondary disabled:opacity-35"
          >
            <Icon icon={ArrowLeft01Icon} size={14} />
          </button>
          <span className="flex items-center gap-1.5">
            {questions.map(({ id: questionId }, i) => (
              <button
                key={questionId}
                type="button"
                aria-label={`Go to question ${i + 1}`}
                aria-current={i === current ? "step" : undefined}
                onClick={() => setStep(i)}
                className={
                  "rounded-full transition-all duration-300 " +
                  (i === current
                    ? "size-2.5 border-2 border-action"
                    : i < current
                      ? // TODO(tokens): no fitting role for bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
                        "size-2 bg-zinc-400"
                      : "size-2 border border-border-strong")
                }
              />
            ))}
          </span>
          <button
            type="button"
            aria-label="Next question"
            disabled={isLast}
            onClick={() => setStep(current + 1)}
            className="flex size-6 items-center justify-center rounded-lg text-foreground-subtle transition-colors enabled:hover:bg-surface-hover enabled:hover:text-foreground-secondary disabled:opacity-35"
          >
            <Icon icon={ArrowRight01Icon} size={14} />
          </button>
        </span>

        <button
          type="button"
          aria-label={isLast ? "Send answers" : "Next question"}
          disabled={!actionEnabled}
          onClick={handleAction}
          className={
            "flex size-8 items-center justify-center rounded-full transition-all duration-200 enabled:active:scale-95 " +
            (actionEnabled
              ? "bg-action text-action-foreground hover:bg-action-hover"
              : "bg-surface-hover text-foreground-subtle")
          }
        >
          <Icon icon={isLast ? SentIcon : ArrowRight01Icon} size={15} />
        </button>
      </div>
    </>
  );
}
