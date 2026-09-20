"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { isKey } from "@/lib/keyboard";
import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import type { MultiAnswer } from "../../tools/clarifying-questions";
import {
  QuestionCheckboxList,
  type QuestionCheckboxListHandle,
} from "./QuestionCheckboxList";

interface Props {
  options: string[];
  value: MultiAnswer;
  labelId: string;
  autoFocus: boolean;
  onChange: (value: MultiAnswer) => void;
  onSubmit: () => void;
}

export interface QuestionMultiAnswerFieldHandle {
  focus: () => void;
}

/** The answer input for a question that takes several answers: the options
 *  as a checkbox group, plus typed text kept *alongside* the ticks rather
 *  than replacing them — "any of these, and also…" is the whole point. */
export const QuestionMultiAnswerField = forwardRef<
  QuestionMultiAnswerFieldHandle,
  Props
>(function QuestionMultiAnswerField(
  { options, value, labelId, autoFocus, onChange, onSubmit },
  ref,
) {
  const listRef = useRef<QuestionCheckboxListHandle>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // The ticks and the typed text arrive in separate slots and stay that way:
  // text that happens to equal an option is still the user's own words, so it
  // neither ticks the box nor drains out of the textarea — including after a
  // remount, since the answer itself remembers which was which.
  const selected = value.selected.filter((pick) => options.includes(pick));
  const custom = value.custom;
  const [typing, setTyping] = useState(() => custom.length > 0);
  // Focus follows an explicit toggle. A saved custom answer reopens the
  // textarea too, but must not steal focus when the card pages back to it —
  // that is what the pager's autoFocus is for.
  const [toggled, setToggled] = useState(false);

  // "Answer this one" lands wherever the user left off: the open textarea if
  // they were typing, otherwise the checkbox group.
  useImperativeHandle(ref, () => ({
    focus: () => {
      if (typing) textareaRef.current?.focus();
      else listRef.current?.focus();
    },
  }));

  function commit(nextSelected: string[], nextCustom: string) {
    // Rebuilt in the order they were offered, so the reply reads like the
    // question however the user ticked their way down it.
    onChange({
      selected: options.filter((option) => nextSelected.includes(option)),
      custom: nextCustom,
    });
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
        ref={listRef}
        options={options}
        selected={selected}
        labelId={labelId}
        focusActiveOption={autoFocus && !typing}
        onToggle={handleToggle}
      />
      {typing ? (
        <textarea
          ref={textareaRef}
          rows={2}
          aria-labelledby={labelId}
          autoFocus={autoFocus || toggled}
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
          onClick={() => {
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
});
