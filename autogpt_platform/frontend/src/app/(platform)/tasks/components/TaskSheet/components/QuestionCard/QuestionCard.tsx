"use client";

import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import type { FormEvent } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
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
      className="flex flex-col gap-3 rounded-xl border border-warning-border bg-warning-surface p-4"
    >
      <div className="flex items-center gap-2">
        <Icon icon={AlertCircleIcon} tone="warning-bright" />
        <Text variant="small-semibold" tone="warning">
          Needs you · {question.asked_by.name} asks
        </Text>
      </div>
      <Text variant="body-tight" tone="primary" className="whitespace-pre-wrap">
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
          <Input
            bare
            aria-label="Your answer"
            placeholder="Type your answer"
            className="flex-1"
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
