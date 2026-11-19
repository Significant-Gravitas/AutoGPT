"use client";

import { cn } from "@/lib/utils";
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
  /** An option was chosen as the answer, to go on with at once. */
  onPick: (value: string) => void;
  onSubmit: () => void;
}

const FIELD =
  "w-full resize-none rounded-lg bg-surface-wash px-2.5 py-2 text-body-sm leading-relaxed text-foreground-ink ring-1 ring-border-hairline transition-shadow duration-150 ease-out ring-inset placeholder:text-foreground-quiet focus:ring-2 focus:ring-focus-inbox focus:outline-none";

/** The answer input for one question: its options as numbered rows, then a
 *  last row that takes the user's own words; a question without options is
 *  that row alone. An option can be copied into the row to reword it or to
 *  build one answer out of several.
 *  Mounted per-question — the parent keys this component on the question's
 *  pager id, so nothing typed leaks from one question to the next. */
export function QuestionAnswerField({
  question,
  value,
  labelId,
  autoFocus,
  onChange,
  onPick,
  onSubmit,
}: Props) {
  const options = question.options ?? [];
  const isCustom = value.trim().length > 0 && !options.includes(value.trim());
  // The row shows only the user's own words, never an option echoed back.
  const [draftFocus, setDraftFocus] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // A second edit adds to the draft rather than replacing it, so the user
  // can build one answer out of several options.
  function handleOptionEdit(option: string) {
    onChange(isCustom ? `${value.trimEnd()}\n${option}` : option);
    setDraftFocus(true);
    textareaRef.current?.focus();
  }

  const ownWords = (
    <textarea
      ref={textareaRef}
      required
      rows={options.length > 0 ? 1 : 3}
      aria-labelledby={labelId}
      autoFocus={options.length === 0 ? autoFocus : draftFocus}
      value={options.length > 0 && !isCustom && !draftFocus ? "" : value}
      onChange={(e) => onChange(e.target.value)}
      onFocus={() => {
        // Starting to type drops a picked option: the row is now the answer.
        if (options.includes(value.trim())) onChange("");
        setDraftFocus(true);
      }}
      // Enter advances the pager; Shift+Enter is the newline.
      onKeyDown={(e) => {
        if (!isKey(e, "Enter") || e.shiftKey) return;
        e.preventDefault();
        onSubmit();
      }}
      // The example is the options joined, so replaying it right under
      // those same options would only be noise.
      placeholder={
        options.length > 0
          ? "Something else…"
          : question.example
            ? `e.g. ${question.example}`
            : "Type your answer"
      }
      className={cn(FIELD, options.length > 0 && "min-h-9")}
    />
  );

  if (options.length === 0) return <div className="px-1 pb-1">{ownWords}</div>;

  return (
    <div className="flex flex-col">
      <QuestionOptionList
        options={options}
        value={value}
        labelId={labelId}
        focusActiveOption={!draftFocus && autoFocus}
        recommended={question.recommended}
        onChange={(option) => {
          setDraftFocus(false);
          onChange(option);
        }}
        onEdit={handleOptionEdit}
        onPick={(option) => {
          setDraftFocus(false);
          onPick(option);
        }}
        onSubmit={onSubmit}
      />
      <div className="px-1 pt-1">{ownWords}</div>
    </div>
  );
}
