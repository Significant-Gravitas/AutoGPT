"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Message01Icon } from "@hugeicons/core-free-icons";
import type { QuestionRequest } from "./helpers";

interface Props {
  requests: QuestionRequest[];
}

export function QuestionsSection({ requests }: Props) {
  const questions = requests.flatMap((request) =>
    request.questions.map((question) => ({ request, question })),
  );
  if (questions.length === 0) return null;

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
          onClick={() => requests.forEach((request) => request.onSkip())}
          className="rounded-full px-2 py-0.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600"
        >
          Skip
        </button>
      </div>

      <div className="flex flex-col gap-3 px-4 py-3">
        {questions.map(({ request, question }) => (
          <label key={question.keyword} className="flex flex-col gap-1.5">
            <span className="text-sm text-zinc-700">{question.question}</span>
            <input
              type="text"
              required
              value={request.answers[question.keyword] ?? ""}
              onChange={(e) =>
                request.onAnswer(question.keyword, e.target.value)
              }
              placeholder={
                question.example
                  ? `e.g. ${question.example}`
                  : "Type your answer"
              }
              className="rounded-2xl bg-zinc-50 px-3 py-2 text-sm text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
            />
          </label>
        ))}
      </div>
    </>
  );
}
