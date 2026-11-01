"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useContext, useEffect, useId, useRef, useState } from "react";
import {
  isAnswered,
  toAnswerText,
  toMultiAnswer,
  type QuestionAnswer,
} from "../../tools/clarifying-questions";
import { QuestionMultiAnswerField } from "../ChainActionCard/QuestionMultiAnswerField";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  ChainActionsContext,
  type ChainActionEntry,
} from "../ToolChain/chainActions";
import {
  buildAnswersMessage,
  buildQuestionsEntry,
  getPendingQuestions,
  type PendingQuestions,
} from "./helpers";
import { isKey } from "@/lib/keyboard";
import { Text } from "@/components/atoms/Text/Text";

interface FormProps {
  dockId: string;
  questions: PendingQuestions["questions"];
}

/** The clarifying-question answer form — rendered inline in the tool chain
 *  on the ask_question row (and reused by the legacy QuestionDock). */
export function QuestionsForm({ dockId, questions }: FormProps) {
  const actions = useContext(CopilotChatActionsContext);
  const chainActions = useContext(ChainActionsContext);
  const [answers, setAnswers] = useState<Record<string, QuestionAnswer>>({});
  const [dismissedId, setDismissedId] = useState<string | null>(null);
  const [renderedDockId, setRenderedDockId] = useState<string | null>(null);
  const inputRefs = useRef<Record<string, { focus: () => void } | null>>({});
  const formId = useId();
  const owner = useId();

  const dismissed = dockId === dismissedId;
  const allAnswered = questions.every((q) => isAnswered(answers[q.keyword]));

  // Inside a tool chain the Answer button is replaced by the chain's single
  // Proceed step — register readiness + message instead.
  useEffect(() => {
    if (!chainActions || dismissed) return;
    function register(entry: ChainActionEntry) {
      chainActions?.register(owner, entry);
    }
    register(
      buildQuestionsEntry({
        dockId,
        questions,
        answers,
        register,
        setAnswers,
        dismiss: () => setDismissedId(dockId),
      }),
    );
    return () => chainActions.unregister(owner);
  }, [chainActions, dismissed, dockId, answers, questions, owner]);

  if (renderedDockId !== dockId) {
    setRenderedDockId(dockId);
    setAnswers({});
  }

  if (!actions || dismissed) return null;
  // Inside a chain the inputs live in the action card below it, not on the
  // row that asked.
  if (chainActions) return null;

  function handleSubmit() {
    if (chainActions) return;
    if (!actions) return;
    if (!allAnswered) {
      const unanswered = questions.find((q) => !isAnswered(answers[q.keyword]));
      if (unanswered) inputRefs.current[unanswered.keyword]?.focus();
      return;
    }
    actions.onSend(buildAnswersMessage(questions, answers));
    setDismissedId(dockId);
  }

  return (
    <div
      // oxlint-disable-next-line design-system/no-arbitrary-values -- the floating card lift shadow; no shadow token has it
      className="animate-fade-up rounded-2xl bg-surface p-3 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)] ring-1 ring-border-base/70 motion-reduce:animate-none"
    >
      <div className="flex items-center justify-between px-0.5 pb-2">
        <Text variant="small-medium-tight" as="span" tone="muted">
          {questions.length === 1 ? "Quick question" : "Quick questions"}
        </Text>
        <button
          type="button"
          onClick={() => setDismissedId(dockId)}
          className="rounded-full px-2 py-0.5 text-xs text-foreground-subtle transition-colors hover:bg-surface-hover hover:text-foreground-secondary"
        >
          Skip
        </button>
      </div>
      <div className="flex flex-col gap-2.5">
        {questions.map((q) =>
          q.allow_multiple && q.options ? (
            <div key={q.keyword} className="flex flex-col gap-1">
              <span
                id={`${formId}-${q.keyword}`}
                className="text-body-sm leading-inherit text-foreground-body"
              >
                {q.question}
              </span>
              <QuestionMultiAnswerField
                ref={(element) => {
                  inputRefs.current[q.keyword] = element;
                }}
                options={q.options}
                value={toMultiAnswer(answers[q.keyword])}
                labelId={`${formId}-${q.keyword}`}
                autoFocus={false}
                onChange={(value) =>
                  setAnswers((prev) => ({ ...prev, [q.keyword]: value }))
                }
                onSubmit={handleSubmit}
              />
            </div>
          ) : (
            <label key={q.keyword} className="flex flex-col gap-1">
              <span className="text-body-sm leading-inherit text-foreground-body">
                {q.question}
              </span>
              <input
                ref={(element) => {
                  inputRefs.current[q.keyword] = element;
                }}
                type="text"
                required
                value={toAnswerText(answers[q.keyword])}
                onChange={(e) =>
                  setAnswers((prev) => ({
                    ...prev,
                    [q.keyword]: e.target.value,
                  }))
                }
                onKeyDown={(e) => {
                  if (isKey(e, "Enter")) handleSubmit();
                }}
                placeholder={
                  q.example ? `e.g. ${q.example}` : "Type your answer"
                }
                className="rounded-xl bg-surface-muted px-2.5 py-1.5 text-body-sm leading-inherit text-foreground-emphasis ring-1 ring-border-base/70 transition-shadow placeholder:text-foreground-subtle focus:ring-foreground-subtle focus:outline-none"
              />
            </label>
          ),
        )}
      </div>
      {!chainActions && (
        <div className="flex justify-end pt-2.5">
          <button
            type="button"
            onClick={handleSubmit}
            aria-disabled={!allAnswered}
            className="rounded-full bg-action-hover px-3.5 py-1.5 text-xs font-medium text-action-foreground transition-opacity hover:bg-action aria-disabled:opacity-40"
          >
            Answer
          </button>
        </div>
      )}
    </div>
  );
}

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
}

export function QuestionDock({ messages }: Props) {
  const pending = getPendingQuestions(messages);
  if (!pending) return null;
  return (
    <div className="mb-2">
      <QuestionsForm
        key={pending.dockId}
        dockId={pending.dockId}
        questions={pending.questions}
      />
    </div>
  );
}
