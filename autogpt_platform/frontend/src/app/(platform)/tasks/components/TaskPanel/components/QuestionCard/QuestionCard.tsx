"use client";

import type { FormEvent } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { TaskQuestion } from "../../../../data/types";
import { useQuestionCard } from "./useQuestionCard";

interface Props {
  taskId: string;
  question: TaskQuestion;
}

export function QuestionCard({ taskId, question }: Props) {
  const { answer, setAnswer, send, isSending, pendingAnswer, isError } =
    useQuestionCard(taskId, question.id);
  const options = question.options ?? [];

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    send(answer);
  }

  return (
    <section
      aria-label="Question for you"
      className="flex animate-inbox-enter flex-col gap-3 rounded-card bg-surface-wash p-4"
    >
      <div className="flex items-center gap-2">
        <span
          aria-hidden
          className="size-2 shrink-0 rounded-full bg-status-waiting"
        />
        <Text variant="inbox-medium" tone="waiting">
          Needs you
        </Text>
        <Text variant="inbox" tone="quiet">
          · {question.asked_by.name} asks
        </Text>
      </div>
      <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
        {question.question}
      </Text>
      {options.length ? (
        <div className="flex flex-wrap gap-2">
          {options.map((option) => (
            <Button
              key={option}
              variant="secondary"
              size="xs"
              disabled={isSending}
              loading={pendingAnswer === option}
              onClick={() => send(option)}
            >
              {option}
            </Button>
          ))}
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            aria-label="Your answer"
            placeholder="Type your answer"
            className="h-8 w-full min-w-0 flex-1 rounded-control bg-surface px-2.5 text-body-sm text-foreground-ink shadow-card transition-shadow duration-150 ease-out-expo placeholder:text-foreground-faint focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
            disabled={isSending}
          />
          <Button
            type="submit"
            variant="primary"
            size="small"
            loading={isSending}
            disabled={!answer.trim()}
          >
            Send answer
          </Button>
        </form>
      )}
      {isError ? (
        <Text variant="small" tone="error">
          Your answer could not be sent. Try again.
        </Text>
      ) : null}
    </section>
  );
}
