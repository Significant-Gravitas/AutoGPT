"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";
import { m } from "framer-motion";
import { useEffect, useId, useRef, useState } from "react";
import {
  isAnswered,
  toAnswerText,
  toMultiAnswer,
} from "../../tools/clarifying-questions";
import type { QuestionRequest } from "./helpers";
import { QuestionAnswerField } from "./QuestionAnswerField";
import { QuestionMultiAnswerField } from "./QuestionMultiAnswerField";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  requests: QuestionRequest[];
  isReady: boolean;
  onProceed: () => void;
}

/** One question per step, in a card that docks above the composer. A click
 *  on an option (or its number) answers at once and moves on; the last row
 *  takes the user's own words. The quiet "1 of 2" pager moves between
 *  questions, Later puts them all off, and the last answer sends every
 *  answer as one message. */
export function QuestionsSection({ requests, isReady, onProceed }: Props) {
  const [step, setStep] = useState(0);
  // A pick on the last question sends a beat later: by then its answer has
  // reached the chain, and the row has had time to show as chosen.
  const latest = useRef({ isReady, onProceed });
  const sendTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    latest.current = { isReady, onProceed };
  });
  useEffect(
    () => () => {
      if (sendTimer.current) clearTimeout(sendTimer.current);
    },
    [],
  );
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

  // Choosing an option is the answer: go straight on.
  function handlePick(value: string) {
    request.onAnswer(question.keyword, value);
    if (!isLast) {
      setStep(current + 1);
      return;
    }
    if (sendTimer.current) clearTimeout(sendTimer.current);
    sendTimer.current = setTimeout(() => {
      if (latest.current.isReady) latest.current.onProceed();
    }, SEND_AFTER_PICK_MS);
  }

  return (
    <div className="rounded-composer-tray bg-surface-tray p-1">
      <div className="flex h-8 items-center gap-2 pr-1 pl-2.5">
        <Text as="span" variant="inbox" tone="quiet" className="truncate">
          {questions.length === 1 ? "A question for you" : "Questions for you"}
        </Text>
        <span className="ml-auto flex shrink-0 items-center gap-0.5">
          {questions.length > 1 ? (
            <>
              <button
                type="button"
                aria-label="Previous question"
                disabled={current === 0}
                onClick={() => setStep(current - 1)}
                className={PAGER}
              >
                <Icon icon={ArrowLeft01Icon} size={13} tone="inherit" />
              </button>
              <Text
                as="span"
                variant="inbox"
                tone="quiet"
                tabular
                className="mx-0.5"
              >
                {current + 1} of {questions.length}
              </Text>
              <button
                type="button"
                aria-label="Next question"
                disabled={isLast}
                onClick={() => setStep(current + 1)}
                className={PAGER}
              >
                <Icon icon={ArrowRight01Icon} size={13} tone="inherit" />
              </button>
            </>
          ) : null}
          <button
            type="button"
            onClick={() => requests.forEach((req) => req.onSkip())}
            className="ml-1 flex h-6 items-center rounded-md px-1.5 text-body-sm text-foreground-quiet transition-colors duration-100 outline-none hover:bg-surface-wash hover:text-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox"
          >
            Later
          </button>
        </span>
      </div>

      <div className="rounded-panel bg-surface shadow-(--inbox-shadow-card)">
        <m.div
          key={id}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
          className="flex flex-col gap-1 p-1"
        >
          <span
            id={labelId}
            className="px-2 pt-2 pb-1 text-body font-medium text-pretty text-foreground-ink"
          >
            {question.question}
          </span>
          {multiOptions.length > 0 ? (
            <div className="px-1 pb-1">
              <QuestionMultiAnswerField
                key={id}
                options={multiOptions}
                value={toMultiAnswer(answer)}
                labelId={labelId}
                autoFocus={current > 0}
                onChange={(value) => request.onAnswer(question.keyword, value)}
                onSubmit={handleAction}
              />
            </div>
          ) : (
            <QuestionAnswerField
              // The field's own state must reset per question; keying it here
              // rather than only on the wrapper keeps that contract local to
              // the component that owns the state.
              key={id}
              question={question}
              value={toAnswerText(answer)}
              labelId={labelId}
              autoFocus={current > 0}
              onChange={(value) => request.onAnswer(question.keyword, value)}
              onPick={handlePick}
              onSubmit={handleAction}
            />
          )}
        </m.div>
        <div className="flex justify-end px-1.5 pb-1.5">
          <button
            type="button"
            aria-label={isLast ? "Send answers" : "Next question"}
            disabled={!actionEnabled}
            onClick={handleAction}
            className={
              "flex size-7 items-center justify-center rounded-full transition-transform duration-150 ease-out outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox enabled:active:scale-95 motion-reduce:transition-none " +
              (actionEnabled
                ? "bg-foreground-ink text-surface"
                : "bg-surface-wash text-foreground-faint")
            }
          >
            <Icon icon={isLast ? SentIcon : ArrowRight01Icon} size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}

const SEND_AFTER_PICK_MS = 120;

const PAGER =
  "flex size-6 items-center justify-center rounded-md text-foreground-quiet transition-colors duration-100 outline-none enabled:hover:bg-surface-wash enabled:hover:text-foreground-ink focus-visible:ring-2 focus-visible:ring-focus-inbox disabled:opacity-35";
