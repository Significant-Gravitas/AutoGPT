"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useContext, useEffect, useRef, useState } from "react";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { ChainActionsContext } from "../ToolChain/chainActions";
import {
  buildAnswersMessage,
  getPendingQuestions,
  type PendingQuestions,
} from "./helpers";

interface FormProps {
  dockId: string;
  questions: PendingQuestions["questions"];
}

/** The clarifying-question answer form — rendered inline in the tool chain
 *  on the ask_question row (and reused by the legacy QuestionDock). */
export function QuestionsForm({ dockId, questions }: FormProps) {
  const actions = useContext(CopilotChatActionsContext);
  const chainActions = useContext(ChainActionsContext);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [dismissedId, setDismissedId] = useState<string | null>(null);
  const [renderedDockId, setRenderedDockId] = useState<string | null>(null);
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const dismissed = dockId === dismissedId;
  const allAnswered = questions.every((q) => answers[q.keyword]?.trim());

  // Inside a tool chain the Answer button is replaced by the chain's single
  // Proceed step — register readiness + message instead.
  useEffect(() => {
    if (!chainActions || dismissed) return;
    chainActions.register({
      id: dockId,
      ready: allAnswered,
      buildMessage: () =>
        allAnswered ? buildAnswersMessage(questions, answers) : null,
      onSent: () => setDismissedId(dockId),
      questions: {
        id: dockId,
        questions,
        answers,
        onAnswer: (keyword, value) =>
          setAnswers((prev) => ({ ...prev, [keyword]: value })),
        onSkip: () => setDismissedId(dockId),
      },
    });
    return () => chainActions.unregister(dockId);
  }, [chainActions, dismissed, dockId, allAnswered, answers, questions]);

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
      const unanswered = questions.find((q) => !answers[q.keyword]?.trim());
      if (unanswered) inputRefs.current[unanswered.keyword]?.focus();
      return;
    }
    actions.onSend(buildAnswersMessage(questions, answers));
    setDismissedId(dockId);
  }

  return (
    <div className="animate-fade-up rounded-2xl bg-white p-3 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)] ring-1 ring-zinc-200/70 motion-reduce:animate-none">
      <div className="flex items-center justify-between px-0.5 pb-2">
        <span className="text-xs font-medium text-zinc-500">
          {questions.length === 1 ? "Quick question" : "Quick questions"}
        </span>
        <button
          type="button"
          onClick={() => setDismissedId(dockId)}
          className="rounded-full px-2 py-0.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600"
        >
          Skip
        </button>
      </div>
      <div className="flex flex-col gap-2.5">
        {questions.map((q) => (
          <label key={q.keyword} className="flex flex-col gap-1">
            <span className="text-[13px] text-zinc-700">{q.question}</span>
            <input
              ref={(element) => {
                inputRefs.current[q.keyword] = element;
              }}
              type="text"
              required
              value={answers[q.keyword] ?? ""}
              onChange={(e) =>
                setAnswers((prev) => ({
                  ...prev,
                  [q.keyword]: e.target.value,
                }))
              }
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
              }}
              placeholder={q.example ? `e.g. ${q.example}` : "Type your answer"}
              className="rounded-xl bg-zinc-50 px-2.5 py-1.5 text-[13px] text-zinc-800 ring-1 ring-zinc-200/70 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-400"
            />
          </label>
        ))}
      </div>
      {!chainActions && (
        <div className="flex justify-end pt-2.5">
          <button
            type="button"
            onClick={handleSubmit}
            aria-disabled={!allAnswered}
            className="rounded-full bg-zinc-900 px-3.5 py-1.5 text-xs font-medium text-white transition-opacity aria-disabled:opacity-40 hover:bg-zinc-700"
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
