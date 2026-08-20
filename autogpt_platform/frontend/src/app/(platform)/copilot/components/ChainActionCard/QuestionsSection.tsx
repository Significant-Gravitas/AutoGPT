"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Message01Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import { m } from "framer-motion";
import { useState } from "react";
import type { QuestionRequest } from "./helpers";

interface Props {
  requests: QuestionRequest[];
  isReady: boolean;
  onProceed: () => void;
}

/** One question per step. The footer pager (chevrons + ring dots) moves
 *  between questions; the round action button advances and, on the last
 *  step, drafts every answer into the chat input. */
export function QuestionsSection({ requests, isReady, onProceed }: Props) {
  const [step, setStep] = useState(0);
  const questions = requests.flatMap((request) =>
    request.questions.map((question) => ({ request, question })),
  );
  if (questions.length === 0) return null;

  const current = Math.min(step, questions.length - 1);
  const { request, question } = questions[current];
  const answered = (request.answers[question.keyword] ?? "").trim().length > 0;
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
      <div className="flex items-center justify-between gap-2.5 border-b border-zinc-100 px-4 py-3">
        <span className="flex items-center gap-2.5">
          <Icon icon={Message01Icon} size={18} className="text-zinc-400" />
          <span className="text-sm font-medium text-zinc-900">
            Answer a few questions
          </span>
        </span>
        <button
          type="button"
          onClick={() => requests.forEach((req) => req.onSkip())}
          className="rounded-full px-2 py-0.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600"
        >
          Skip
        </button>
      </div>

      <m.div
        key={question.keyword}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
        className="flex flex-col gap-1.5 px-4 py-3"
      >
        <label className="flex flex-col gap-1.5">
          <span className="text-sm text-zinc-700">{question.question}</span>
          <input
            type="text"
            required
            autoFocus={current > 0}
            value={request.answers[question.keyword] ?? ""}
            onChange={(e) => request.onAnswer(question.keyword, e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAction()}
            placeholder={
              question.example ? `e.g. ${question.example}` : "Type your answer"
            }
            className="rounded-2xl bg-zinc-50 px-3 py-2 text-sm text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
          />
        </label>
      </m.div>

      <div className="flex items-center justify-between px-4 pb-3 pt-1">
        <span className="flex items-center gap-2">
          <button
            type="button"
            aria-label="Previous question"
            disabled={current === 0}
            onClick={() => setStep(current - 1)}
            className="flex size-6 items-center justify-center rounded-lg text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:opacity-35"
          >
            <Icon icon={ArrowLeft01Icon} size={14} />
          </button>
          <span className="flex items-center gap-1.5">
            {questions.map(({ question: q }, i) => (
              <button
                key={q.keyword}
                type="button"
                aria-label={`Go to question ${i + 1}`}
                aria-current={i === current ? "step" : undefined}
                onClick={() => setStep(i)}
                className={
                  "rounded-full transition-all duration-300 " +
                  (i === current
                    ? "size-2.5 border-2 border-zinc-800"
                    : i < current
                      ? "size-2 bg-zinc-400"
                      : "size-2 border border-zinc-300")
                }
              />
            ))}
          </span>
          <button
            type="button"
            aria-label="Next question"
            disabled={isLast}
            onClick={() => setStep(current + 1)}
            className="flex size-6 items-center justify-center rounded-lg text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:opacity-35"
          >
            <Icon icon={ArrowRight01Icon} size={14} />
          </button>
        </span>

        <button
          type="button"
          aria-label={isLast ? "Add answers to message" : "Next question"}
          disabled={!actionEnabled}
          onClick={handleAction}
          className={
            "flex size-8 items-center justify-center rounded-full transition-all duration-200 enabled:active:scale-95 " +
            (actionEnabled
              ? "bg-zinc-800 text-white hover:bg-zinc-900"
              : "bg-zinc-100 text-zinc-400")
          }
        >
          <Icon icon={isLast ? SentIcon : ArrowRight01Icon} size={15} />
        </button>
      </div>
    </>
  );
}
