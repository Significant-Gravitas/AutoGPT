"use client";

import { CheckCircleIcon } from "@phosphor-icons/react";
import { useContext, useState } from "react";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  type ClarifyingQuestion,
  normalizeClarifyingQuestions,
} from "../../tools/clarifying-questions";
import type { ChainRow } from "./helpers";
import { QuestionsCard } from "./InfoCards";
import { CARD } from "./ResultCards";
import { asItems, asObject } from "./resultHelpers";

function getQuestions(row: ChainRow): ClarifyingQuestion[] {
  const output = asObject(row.output);
  const input = asObject(row.input);
  const raw =
    (output && asItems(output.questions)) ??
    (input && asItems(input.questions)) ??
    [];
  return normalizeClarifyingQuestions(
    raw as Array<{ question: string; keyword: string; example?: unknown }>,
  );
}

export function QuestionRowForm({ row }: { row: ChainRow }) {
  const actions = useContext(CopilotChatActionsContext);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [sent, setSent] = useState(false);
  const questions = getQuestions(row);

  if (questions.length === 0) return null;
  if (!actions)
    return (
      <QuestionsCard
        questions={questions as unknown as Record<string, unknown>[]}
      />
    );

  if (sent) {
    return (
      <div className={CARD + " flex flex-col gap-1.5 p-2.5"}>
        {questions.map((q) => (
          <div key={q.keyword} className="flex items-start gap-2 text-[13px]">
            <CheckCircleIcon
              size={15}
              weight="fill"
              className="mt-0.5 shrink-0 text-green-500"
            />
            <div className="min-w-0">
              <p className="text-zinc-500">{q.question}</p>
              <p className="font-medium text-zinc-800">
                {answers[q.keyword] || "—"}
              </p>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const allAnswered = questions.every((q) => answers[q.keyword]?.trim());

  function handleSubmit() {
    if (!allAnswered || !actions) return;
    const message = questions
      .map((q) => `> ${q.question}\n\n${answers[q.keyword].trim()}`)
      .join("\n\n");
    actions.onSend(`**Here are my answers:**\n\n${message}\n\nPlease proceed.`);
    setSent(true);
  }

  return (
    <div className={CARD + " flex flex-col gap-2.5 p-3"}>
      {questions.map((q) => (
        <label key={q.keyword} className="flex flex-col gap-1">
          <span className="text-[13px] text-zinc-700">{q.question}</span>
          <input
            type="text"
            value={answers[q.keyword] ?? ""}
            onChange={(e) =>
              setAnswers((prev) => ({ ...prev, [q.keyword]: e.target.value }))
            }
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSubmit();
            }}
            placeholder={q.example ? `e.g. ${q.example}` : "Type your answer"}
            className="rounded-lg bg-zinc-50 px-2.5 py-1.5 text-[13px] text-zinc-800 ring-1 ring-zinc-200/70 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-400"
          />
        </label>
      ))}
      <button
        type="button"
        onClick={handleSubmit}
        disabled={!allAnswered}
        className="self-end rounded-full bg-zinc-900 px-3.5 py-1.5 text-xs font-medium text-white transition-opacity hover:bg-zinc-700 disabled:opacity-40"
      >
        Proceed
      </button>
    </div>
  );
}
