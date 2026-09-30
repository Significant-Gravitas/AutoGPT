"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { isKey } from "@/lib/keyboard";
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

interface Props {
  /** Expected trimmed and deduped — `toOptions` establishes that invariant,
   *  and the option strings double as React keys and selection identities. */
  options: string[];
  selected: string[];
  labelId: string;
  focusActiveOption: boolean;
  onToggle: (option: string) => void;
}

export interface QuestionCheckboxListHandle {
  focus: () => void;
}

/** The options for a multi-select question as a real checkbox group: one stop
 *  in the tab order, arrows moving focus only. The radiogroup's "arrows also
 *  select" model cannot work here — arrowing past four boxes would tick all
 *  four on the way to the fifth. */
export const QuestionCheckboxList = forwardRef<
  QuestionCheckboxListHandle,
  Props
>(function QuestionCheckboxList(
  { options, selected, labelId, focusActiveOption, onToggle },
  ref,
) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const first = options.findIndex((option) => selected.includes(option));
  const active = first === -1 ? 0 : first;

  useImperativeHandle(ref, () => ({
    focus: () => refs.current[active]?.focus(),
  }));

  // Focus lands on the group once, when asked. `active` is the tab stop, not
  // the caret: it follows the first tick, and re-running on every change would
  // yank focus off the box the user just unticked.
  const focused = useRef(false);
  useEffect(() => {
    if (!focusActiveOption) {
      focused.current = false;
      return;
    }
    if (focused.current) return;
    focused.current = true;
    refs.current[active]?.focus();
  }, [focusActiveOption, active]);

  function handleKeyDown(event: React.KeyboardEvent, index: number) {
    // Enter toggles rather than submits: with several picks to make, an Enter
    // that sent the answer would cut the user off after their first box.
    if (isKey(event, "Enter")) {
      event.preventDefault();
      onToggle(options[index]);
      return;
    }
    const step = isKey(event, "ArrowDown", "ArrowRight")
      ? 1
      : isKey(event, "ArrowUp", "ArrowLeft")
        ? -1
        : 0;
    if (step === 0) return;
    event.preventDefault();
    refs.current[(index + step + options.length) % options.length]?.focus();
  }

  return (
    <div role="group" aria-labelledby={labelId} className="flex flex-col gap-2">
      {options.map((option, index) => {
        const isSelected = selected.includes(option);
        return (
          <button
            key={option}
            ref={(element) => {
              refs.current[index] = element;
            }}
            type="button"
            role="checkbox"
            aria-checked={isSelected}
            tabIndex={index === active ? 0 : -1}
            onClick={() => onToggle(option)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            className={
              "flex items-center justify-between gap-3 rounded-2xl px-4 py-3 text-left text-base leading-snug transition-all " +
              (isSelected
                ? "bg-white text-zinc-900 ring-2 ring-zinc-800"
                : "bg-zinc-50 text-zinc-700 ring-1 ring-zinc-100 hover:bg-zinc-100")
            }
          >
            <span>{option}</span>
            <span
              className={
                "flex size-5 shrink-0 items-center justify-center rounded-md transition-colors " +
                (isSelected
                  ? "bg-zinc-800 text-white"
                  : "ring-1 ring-inset ring-zinc-300")
              }
            >
              {isSelected && <Icon icon={Tick02Icon} size={13} />}
            </span>
          </button>
        );
      })}
    </div>
  );
});
