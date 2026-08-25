"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { ClarifyingQuestion } from "../../tools/clarifying-questions";

interface Props {
  question: ClarifyingQuestion;
  value: string;
  autoFocus: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

/** The answer input for one question: options render as tappable rows with a
 *  trailing "Type something…" escape hatch into free text; questions without
 *  options go straight to the textarea. Mounted per-question (parent keys on
 *  the keyword), so the typing toggle resets naturally between questions. */
export function QuestionAnswerField({
  question,
  value,
  autoFocus,
  onChange,
  onSubmit,
}: Props) {
  const options = question.options ?? [];
  const [typing, setTyping] = useState(
    () =>
      options.length > 0 && value.trim().length > 0 && !options.includes(value),
  );

  if (options.length === 0 || typing) {
    return (
      <div className="flex flex-col gap-1.5">
        <textarea
          required
          rows={3}
          autoFocus={autoFocus || typing}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          // Enter still advances the pager; Shift+Enter is the newline.
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            e.preventDefault();
            onSubmit();
          }}
          placeholder={
            question.example ? `e.g. ${question.example}` : "Type your answer"
          }
          className="resize-none rounded-2xl bg-zinc-50 px-3 py-2 text-sm leading-relaxed text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
        />
        {options.length > 0 && (
          <button
            type="button"
            onClick={() => setTyping(false)}
            className="self-start rounded-full px-2 py-0.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600"
          >
            Choose from options instead
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5" role="radiogroup">
      {options.map((option) => {
        const selected = value === option;
        return (
          <button
            key={option}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(option)}
            className={
              "flex items-center justify-between gap-2 rounded-2xl px-3 py-2 text-left text-sm leading-relaxed transition-all " +
              (selected
                ? "bg-white text-zinc-900 ring-2 ring-zinc-800"
                : "bg-zinc-50 text-zinc-700 ring-1 ring-zinc-100 hover:bg-zinc-100")
            }
          >
            <span>{option}</span>
            {selected && (
              <Icon icon={Tick02Icon} size={14} className="shrink-0" />
            )}
          </button>
        );
      })}
      <button
        type="button"
        onClick={() => {
          if (options.includes(value)) onChange("");
          setTyping(true);
        }}
        className="flex items-center gap-2 rounded-2xl border border-dashed border-zinc-200 px-3 py-2 text-left text-sm text-zinc-500 transition-colors hover:border-zinc-300 hover:text-zinc-700"
      >
        <Icon icon={PencilEdit02Icon} size={14} className="shrink-0" />
        Type something…
      </button>
    </div>
  );
}
