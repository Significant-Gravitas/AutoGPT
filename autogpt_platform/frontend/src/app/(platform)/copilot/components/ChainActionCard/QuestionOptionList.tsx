"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PencilEdit02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useEffect, useRef } from "react";
import { isKey } from "@/lib/keyboard";

interface Props {
  /** Expected trimmed and deduped — `toOptions` establishes that invariant,
   *  and the option strings double as React keys and selection identities. */
  options: string[];
  value: string;
  labelId: string;
  focusActiveOption: boolean;
  onChange: (value: string) => void;
  /** Copies the option into free text, for an answer that is close but not
   *  quite right as written. */
  onEdit: (value: string) => void;
  onSubmit: () => void;
}

/** The options for one question as a real radiogroup: a single stop in the tab
 *  order, with arrows moving and selecting. Anything less would leave the radio
 *  roles promising assistive tech a keyboard model that isn't there. */
export function QuestionOptionList({
  options,
  value,
  labelId,
  focusActiveOption,
  onChange,
  onEdit,
  onSubmit,
}: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const selected = options.indexOf(value.trim());
  const active = selected === -1 ? 0 : selected;

  useEffect(() => {
    if (focusActiveOption) refs.current[active]?.focus();
  }, [focusActiveOption, active]);

  function moveTo(index: number) {
    onChange(options[index]);
    refs.current[index]?.focus();
  }

  function handleKeyDown(event: React.KeyboardEvent, index: number) {
    // Enter would otherwise re-click the focused option and leave the user
    // tabbing past the pager to reach send. Selecting first means tabbing in
    // and hitting Enter can't submit an option nobody chose — and arrowing or
    // clicking already selects, so those reach the pager on the first Enter.
    if (isKey(event, "Enter")) {
      event.preventDefault();
      if (options[index] === value.trim()) onSubmit();
      else onChange(options[index]);
      return;
    }
    if (isKey(event, " ")) {
      event.preventDefault();
      onChange(options[index]);
      return;
    }
    const step = isKey(event, "ArrowDown", "ArrowRight")
      ? 1
      : isKey(event, "ArrowUp", "ArrowLeft")
        ? -1
        : 0;
    if (step === 0) return;
    event.preventDefault();
    moveTo((index + step + options.length) % options.length);
  }

  return (
    <div
      role="radiogroup"
      aria-labelledby={labelId}
      aria-required="true"
      className="flex flex-col gap-2"
    >
      {options.map((option, index) => {
        const isSelected = option === value.trim();
        return (
          <div key={option} className="relative">
            <button
              ref={(element) => {
                refs.current[index] = element;
              }}
              type="button"
              role="radio"
              aria-checked={isSelected}
              tabIndex={index === active ? 0 : -1}
              onClick={() => onChange(option)}
              onKeyDown={(event) => handleKeyDown(event, index)}
              className={
                "flex w-full items-center justify-between gap-3 rounded-2xl py-3 pl-4 pr-12 text-left text-base leading-snug transition-all " +
                (isSelected
                  ? "bg-white text-zinc-900 ring-2 ring-zinc-800"
                  : "bg-zinc-50 text-zinc-700 ring-1 ring-zinc-100 hover:bg-zinc-100")
              }
            >
              <span>{option}</span>
              {isSelected && (
                <Icon icon={Tick02Icon} size={16} className="shrink-0" />
              )}
            </button>
            <button
              type="button"
              aria-label={`Edit ${option}`}
              title="Edit before sending"
              onClick={() => onEdit(option)}
              className="absolute right-2 top-1/2 flex size-8 -translate-y-1/2 items-center justify-center rounded-xl text-zinc-400 transition-colors hover:bg-zinc-200/60 hover:text-zinc-700"
            >
              <Icon icon={PencilEdit02Icon} size={16} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
