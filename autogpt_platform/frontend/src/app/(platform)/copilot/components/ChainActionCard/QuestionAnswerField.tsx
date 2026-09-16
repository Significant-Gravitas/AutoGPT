"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { ClarifyingQuestion } from "../../tools/clarifying-questions";
import { QuestionOptionList } from "./QuestionOptionList";

interface Props {
  question: ClarifyingQuestion;
  value: string;
  labelId: string;
  autoFocus: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

/** The answer input for one question: options render as tappable rows with a
 *  trailing "Type something…" escape hatch into free text; questions without
 *  options go straight to the textarea. Mounted per-question — the parent keys
 *  this component on the question's pager id, so the typing toggle resets
 *  between questions instead of leaking across them. */
export function QuestionAnswerField({
  question,
  value,
  labelId,
  autoFocus,
  onChange,
  onSubmit,
}: Props) {
  const options = question.options ?? [];
  const isCustom = value.trim().length > 0 && !options.includes(value.trim());
  const [typing, setTyping] = useState(() => options.length > 0 && isCustom);
  // Focus follows an explicit toggle. A pre-filled custom answer opens the
  // textarea too, but must not steal focus when the card first renders —
  // that is what the pager's autoFocus is for.
  const [toggled, setToggled] = useState(false);

  if (options.length === 0 || typing) {
    return (
      <div className="flex flex-col gap-1.5">
        <textarea
          required
          rows={3}
          aria-labelledby={labelId}
          autoFocus={autoFocus || toggled}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          // Enter advances the pager; Shift+Enter is the newline.
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            e.preventDefault();
            onSubmit();
          }}
          // The example is the options joined, so replaying it to someone who
          // just declined those options would only be noise.
          placeholder={
            options.length === 0 && question.example
              ? `e.g. ${question.example}`
              : "Type your answer"
          }
          className="resize-none rounded-2xl bg-zinc-50 px-3 py-2 text-sm leading-relaxed text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
        />
        {options.length > 0 && (
          <button
            type="button"
            onClick={() => {
              // Drop the free text, or the pager would happily submit a value
              // the option list gives no sign of having selected.
              if (isCustom) onChange("");
              setToggled(true);
              setTyping(false);
            }}
            className="self-start rounded-full px-2 py-0.5 text-xs text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            Choose from options instead
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <QuestionOptionList
        options={options}
        value={value}
        labelId={labelId}
        focusActiveOption={autoFocus || toggled}
        onChange={onChange}
        onSubmit={onSubmit}
      />
      <button
        type="button"
        onClick={() => {
          if (options.includes(value.trim())) onChange("");
          setToggled(true);
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
