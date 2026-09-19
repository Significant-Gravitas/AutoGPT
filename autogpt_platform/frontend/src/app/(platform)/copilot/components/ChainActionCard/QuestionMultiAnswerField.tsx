"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { isKey } from "@/lib/keyboard";
import { useState } from "react";
import type { QuestionAnswer } from "../../tools/clarifying-questions";
import { QuestionCheckboxList } from "./QuestionCheckboxList";

interface Props {
  options: string[];
  value: QuestionAnswer;
  labelId: string;
  autoFocus: boolean;
  onChange: (value: string[]) => void;
  onSubmit: () => void;
}

/** The answer input for a question that takes several answers: the options
 *  as a checkbox group, plus typed text kept *alongside* the ticks rather
 *  than replacing them — "any of these, and also…" is the whole point. */
export function QuestionMultiAnswerField({
  options,
  value,
  labelId,
  autoFocus,
  onChange,
  onSubmit,
}: Props) {
  // Picks and typed text share one list; what separates them is membership of
  // `options`, the same rule the single-select field uses for a custom answer.
  const picks = Array.isArray(value) ? value : value ? [value] : [];
  const selected = picks.filter((pick) => options.includes(pick));
  const custom = picks.find((pick) => !options.includes(pick)) ?? "";
  const [typing, setTyping] = useState(() => custom.length > 0);

  function commit(nextSelected: string[], nextCustom: string) {
    // Rebuilt in the order they were offered, so the reply reads like the
    // question however the user ticked their way down it.
    onChange([
      ...options.filter((option) => nextSelected.includes(option)),
      ...(nextCustom ? [nextCustom] : []),
    ]);
  }

  function handleToggle(option: string) {
    commit(
      selected.includes(option)
        ? selected.filter((pick) => pick !== option)
        : [...selected, option],
      custom,
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <QuestionCheckboxList
        options={options}
        selected={selected}
        labelId={labelId}
        focusActiveOption={autoFocus}
        onToggle={handleToggle}
      />
      {typing ? (
        <textarea
          rows={2}
          aria-labelledby={labelId}
          autoFocus
          value={custom}
          onChange={(e) => commit(selected, e.target.value)}
          // Enter advances the pager; Shift+Enter is the newline.
          onKeyDown={(e) => {
            if (!isKey(e, "Enter") || e.shiftKey) return;
            e.preventDefault();
            onSubmit();
          }}
          placeholder="Add your own answer"
          className="resize-none rounded-2xl bg-zinc-50 px-4 py-3 text-base leading-relaxed text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
        />
      ) : (
        <button
          type="button"
          onClick={() => setTyping(true)}
          className="flex items-center gap-2.5 rounded-2xl border border-dashed border-zinc-200 px-4 py-3 text-left text-base text-zinc-500 transition-colors hover:border-zinc-300 hover:text-zinc-700"
        >
          <Icon icon={PencilEdit02Icon} size={16} className="shrink-0" />
          Type something…
        </button>
      )}
    </div>
  );
}
