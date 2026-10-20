"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { useRef, useState } from "react";
import type { ClarifyingQuestion } from "../../tools/clarifying-questions";
import { QuestionOptionList } from "./QuestionOptionList";
import { isKey } from "@/lib/keyboard";

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
 *  options go straight to the textarea. The options stay on screen while the
 *  user types, and each one can be copied into the textarea to reword it or
 *  merge it with others.
 *  Mounted per-question — the parent keys this component on the question's
 *  pager id, so the typing toggle resets between questions instead of leaking
 *  across them. */
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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const textarea = (
    <textarea
      ref={textareaRef}
      required
      rows={options.length > 0 ? 2 : 3}
      aria-labelledby={labelId}
      autoFocus={autoFocus || toggled}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      // Enter advances the pager; Shift+Enter is the newline.
      onKeyDown={(e) => {
        if (!isKey(e, "Enter") || e.shiftKey) return;
        e.preventDefault();
        onSubmit();
      }}
      // The example is the options joined, so replaying it right under
      // those same options would only be noise.
      placeholder={
        options.length === 0 && question.example
          ? `e.g. ${question.example}`
          : "Type your answer"
      }
      className="resize-none rounded-2xl bg-zinc-50 px-4 py-3 text-base leading-relaxed text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
    />
  );

  if (options.length === 0) return textarea;

  // Choosing an option replaces whatever was typed, so the textarea closes
  // rather than echoing the option's text back as if the user had written it.
  function handleOptionChange(option: string) {
    setTyping(false);
    onChange(option);
  }

  // A second edit adds to the draft rather than replacing it, so the user
  // can build one answer out of several options.
  function handleOptionEdit(option: string) {
    const draft =
      typing && value.trim() ? `${value.trimEnd()}\n${option}` : option;
    onChange(draft);
    setToggled(true);
    setTyping(true);
    textareaRef.current?.focus();
  }

  return (
    <div className="flex flex-col gap-2">
      <QuestionOptionList
        options={options}
        value={value}
        labelId={labelId}
        focusActiveOption={!typing && (autoFocus || toggled)}
        onChange={handleOptionChange}
        onEdit={handleOptionEdit}
        onSubmit={onSubmit}
      />
      {typing ? (
        textarea
      ) : (
        <button
          type="button"
          onClick={() => {
            if (options.includes(value.trim())) onChange("");
            setToggled(true);
            setTyping(true);
          }}
          className="flex items-center gap-2.5 rounded-2xl border border-dashed border-zinc-200 px-4 py-3 text-left text-base text-zinc-500 transition-colors hover:border-zinc-300 hover:text-zinc-700"
        >
          <Icon icon={PencilEdit02Icon} size={16} className="shrink-0" />
          Type something…
        </button>
      )}
    </div>
  );
}
