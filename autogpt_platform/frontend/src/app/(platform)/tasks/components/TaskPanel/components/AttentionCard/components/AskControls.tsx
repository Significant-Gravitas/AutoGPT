"use client";

import type { Ask } from "@/app/api/__generated__/models/ask";
import { Button } from "@/components/atoms/Button/Button";
import type { FormEvent } from "react";
import { orderOptions } from "../helpers";
import { FIELD } from "./styles";

interface Props {
  ask: Ask;
  answer: string;
  onAnswerChange(value: string): void;
  onSend(value: string, askId?: string): void;
  isAnswering: boolean;
  pendingAnswer?: string;
}

/** The ask's options as buttons that wrap their text, and a free reply. */
export function AskControls({
  ask,
  answer,
  onAnswerChange,
  onSend,
  isAnswering,
  pendingAnswer,
}: Props) {
  const options = orderOptions(ask);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSend(answer, ask.id);
  }

  return (
    <>
      {options.length ? (
        <div className="flex flex-wrap gap-1.5">
          {options.map((option) => (
            <Button
              key={option}
              variant={option === ask.default_option ? "primary" : "secondary"}
              size="xs"
              className="h-auto min-h-7 max-w-full min-w-0 text-left break-words whitespace-normal"
              disabled={isAnswering}
              loading={pendingAnswer === option}
              onClick={() => onSend(option, ask.id)}
            >
              {option}
            </Button>
          ))}
        </div>
      ) : null}
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          type="text"
          aria-label="Your answer"
          placeholder={options.length ? "Or type your own" : "Type your answer"}
          className={FIELD}
          value={answer}
          onChange={(event) => onAnswerChange(event.target.value)}
          disabled={isAnswering}
        />
        <Button
          type="submit"
          variant={options.length ? "secondary" : "primary"}
          size="xs"
          loading={isAnswering && pendingAnswer === answer.trim()}
          disabled={!answer.trim()}
        >
          Send
        </Button>
      </form>
    </>
  );
}
