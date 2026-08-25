"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { useEffect, useRef } from "react";

interface Props {
  options: string[];
  value: string;
  labelId: string;
  autoFocus: boolean;
  onChange: (value: string) => void;
}

/** The options for one question as a real radiogroup: a single stop in the tab
 *  order, with arrows moving and selecting. Anything less would leave the radio
 *  roles promising assistive tech a keyboard model that isn't there. */
export function QuestionOptionList({
  options,
  value,
  labelId,
  autoFocus,
  onChange,
}: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const selected = options.indexOf(value.trim());
  const active = selected === -1 ? 0 : selected;

  useEffect(() => {
    if (autoFocus) refs.current[active]?.focus();
  }, [autoFocus, active]);

  function moveTo(index: number) {
    onChange(options[index]);
    refs.current[index]?.focus();
  }

  function handleKeyDown(event: React.KeyboardEvent, index: number) {
    const step =
      event.key === "ArrowDown" || event.key === "ArrowRight"
        ? 1
        : event.key === "ArrowUp" || event.key === "ArrowLeft"
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
      className="flex flex-col gap-1.5"
    >
      {options.map((option, index) => {
        const isSelected = option === value.trim();
        return (
          <button
            key={option}
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
              "flex items-center justify-between gap-2 rounded-2xl px-3 py-2 text-left text-sm leading-relaxed transition-all " +
              (isSelected
                ? "bg-white text-zinc-900 ring-2 ring-zinc-800"
                : "bg-zinc-50 text-zinc-700 ring-1 ring-zinc-100 hover:bg-zinc-100")
            }
          >
            <span>{option}</span>
            {isSelected && (
              <Icon icon={Tick02Icon} size={14} className="shrink-0" />
            )}
          </button>
        );
      })}
    </div>
  );
}
