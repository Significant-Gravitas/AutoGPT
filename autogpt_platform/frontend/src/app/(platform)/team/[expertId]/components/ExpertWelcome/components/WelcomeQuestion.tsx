"use client";

import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { useId, useState } from "react";

interface Props {
  question: string;
  options: string[];
  answer: string | undefined;
  disabled: boolean;
  onAnswer: (value: string) => void;
}

const CHIP_CLASS =
  "inline-flex min-h-9 items-center rounded-full border px-3.5 py-1.5 text-body leading-5 outline-none transition-[background-color,border-color,color,scale] duration-150 ease-out focus-visible:ring-2 focus-visible:ring-foreground-subtle focus-visible:ring-offset-1 active:scale-[0.96] disabled:pointer-events-none motion-reduce:transition-none [@media(hover:none)]:min-h-11";

/** One opening question: a tap on a chip answers it, "Something else" turns
 *  into a text field for an answer in the user's own words. */
export function WelcomeQuestion({
  question,
  options,
  answer,
  disabled,
  onAnswer,
}: Props) {
  const labelId = useId();
  const isCustom = answer !== undefined && !options.includes(answer);
  const [isTyping, setIsTyping] = useState(false);
  const [text, setText] = useState("");

  function commitText() {
    const value = text.trim();
    if (!value) return;
    setIsTyping(false);
    onAnswer(value);
  }

  return (
    <div className="flex flex-col gap-2">
      <Text variant="body" tone="primary" id={labelId}>
        {question}
      </Text>
      <div
        role="radiogroup"
        aria-labelledby={labelId}
        className="flex flex-wrap gap-1.5"
      >
        {options.map((option) => {
          const isSelected = answer === option;
          return (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={isSelected}
              disabled={disabled}
              onClick={() => onAnswer(option)}
              className={cn(
                CHIP_CLASS,
                isSelected
                  ? "border-action bg-action text-action-foreground"
                  : "border-border-base bg-surface text-foreground-emphasis hover:border-border-strong hover:bg-surface-muted",
              )}
            >
              {option}
            </button>
          );
        })}
        {isCustom ? (
          <button
            type="button"
            role="radio"
            aria-checked
            disabled={disabled}
            onClick={() => {
              setText(answer ?? "");
              setIsTyping(true);
            }}
            className={cn(
              CHIP_CLASS,
              "border-action bg-action text-action-foreground",
            )}
          >
            {answer}
          </button>
        ) : null}
        {!isCustom && !isTyping ? (
          <button
            type="button"
            disabled={disabled}
            onClick={() => setIsTyping(true)}
            className={cn(
              CHIP_CLASS,
              "border-dashed border-border-base text-foreground-tertiary hover:border-border-strong hover:text-foreground-body",
            )}
          >
            Something else
          </button>
        ) : null}
      </div>
      {isTyping ? (
        <input
          autoFocus
          aria-labelledby={labelId}
          value={text}
          enterKeyHint="done"
          placeholder="Type your answer, then press Enter"
          onChange={(event) => setText(event.target.value)}
          onBlur={commitText}
          onKeyDown={(event) => {
            if (isKey(event, "Enter")) {
              event.preventDefault();
              commitText();
            }
            if (isKey(event, "Escape")) setIsTyping(false);
          }}
          // 16px on phones: iOS zooms the page into any smaller text field.
          className="rounded-xl border border-border-base bg-surface px-3.5 py-2 text-base text-foreground-base placeholder:text-placeholder focus:border-brand focus:ring-1 focus:ring-brand focus:outline-none sm:text-body"
        />
      ) : null}
    </div>
  );
}
