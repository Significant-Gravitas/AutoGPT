"use client";

import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useContext, useRef, useState } from "react";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { buildAnswersMessage, getPendingQuestions } from "./helpers";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
}

export function QuestionDock({ messages }: Props) {
  const actions = useContext(CopilotChatActionsContext);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [dismissedId, setDismissedId] = useState<string | null>(null);
  const [renderedDockId, setRenderedDockId] = useState<string | null>(null);
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const pending = getPendingQuestions(messages);

  if (!actions || !pending || pending.dockId === dismissedId) return null;
  const { dockId, questions } = pending;

  if (renderedDockId !== dockId) {
    setRenderedDockId(dockId);
    setAnswers({});
  }

  const allAnswered = questions.every((q) => answers[q.keyword]?.trim());

  function handleSubmit() {
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
    <div className="mb-2 animate-fade-up rounded-2xl bg-white p-3 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)] ring-1 ring-zinc-200/70 motion-reduce:animate-none">
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
    </div>
  );
}
