"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import type { Ref } from "react";
import type { QuestionAnswer } from "../../../../tools/clarifying-questions";
import { getOptionLetter } from "../../helpers";
import {
  useOnboardingChoices,
  type OnboardingChoicesHandle,
} from "./useOnboardingChoices";

interface Props {
  ref?: Ref<OnboardingChoicesHandle>;
  options: string[];
  answer: QuestionAnswer | undefined;
  isMultiple: boolean;
  labelId: string;
  onChoose: (option: string) => void;
  onChange: (answer: QuestionAnswer) => void;
  onSubmit: () => void;
}

/** One question's answers as lettered rows: a tap or its letter key picks
 *  the row, and the last row, "Other", opens a text box. A multi-select
 *  question ticks rows instead and keeps the typed text beside the ticks.
 *  Mounted per step, so the text box toggle starts fresh on every question. */
export function OnboardingChoices({
  ref,
  options,
  answer,
  isMultiple,
  labelId,
  onChoose,
  onChange,
  onSubmit,
}: Props) {
  const {
    active,
    isTyping,
    optionRefs,
    selected,
    text,
    changeText,
    choose,
    startTyping,
    handleOptionKeyDown,
    handleTextKeyDown,
  } = useOnboardingChoices({
    ref,
    options,
    answer,
    isMultiple,
    onChoose,
    onChange,
    onSubmit,
  });

  return (
    <div className="flex flex-col gap-2">
      {options.length > 0 && (
        <div
          role={isMultiple ? "group" : "radiogroup"}
          aria-labelledby={labelId}
          aria-required={isMultiple ? undefined : true}
          className="flex flex-col gap-2"
        >
          {options.map((option, index) => {
            const isSelected = selected.includes(option);
            return (
              <button
                key={option}
                ref={(element) => {
                  optionRefs.current[index] = element;
                }}
                type="button"
                role={isMultiple ? "checkbox" : "radio"}
                aria-checked={isSelected}
                tabIndex={index === active ? 0 : -1}
                autoFocus={!isTyping && index === active}
                onClick={() => choose(option)}
                onKeyDown={(event) => handleOptionKeyDown(event, index)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-md px-3 py-3 text-left text-base leading-snug outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 sm:py-2.5 sm:text-[15px] [@media(hover:none)]:min-h-11",
                  isSelected
                    ? "bg-zinc-900 text-white"
                    : "bg-zinc-50 text-zinc-800 ring-1 ring-zinc-100 hover:bg-zinc-100",
                )}
              >
                <LetterBadge index={index} isSelected={isSelected} />
                <span className="flex-1">{option}</span>
                {isSelected && (
                  <Icon icon={Tick02Icon} size={16} className="shrink-0" />
                )}
              </button>
            );
          })}
        </div>
      )}

      {isTyping ? (
        <textarea
          required={!isMultiple}
          rows={2}
          aria-labelledby={labelId}
          autoFocus
          enterKeyHint="next"
          value={text}
          onChange={(event) => changeText(event.target.value)}
          onKeyDown={handleTextKeyDown}
          placeholder={isMultiple ? "Add your own answer" : "Type your answer"}
          // 16px on phones: iOS zooms the page into any smaller text field.
          className="resize-none rounded-md bg-zinc-50 px-4 py-3 text-base leading-relaxed text-zinc-800 ring-1 ring-zinc-200 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-400 sm:text-[15px]"
        />
      ) : (
        <button
          type="button"
          onClick={startTyping}
          className="flex w-full items-center gap-3 rounded-md border border-dashed border-zinc-200 px-3 py-3 text-left text-base text-zinc-500 outline-none transition-colors hover:bg-zinc-50 hover:text-zinc-700 focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 sm:py-2.5 sm:text-[15px] [@media(hover:none)]:min-h-11"
        >
          <LetterBadge index={options.length} isSelected={false} />
          Other
        </button>
      )}
    </div>
  );
}

interface LetterBadgeProps {
  index: number;
  isSelected: boolean;
}

// Letter keys mean nothing on a touch screen, so neither do their badges.
function LetterBadge({ index, isSelected }: LetterBadgeProps) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-6 shrink-0 items-center justify-center rounded-md text-xs font-medium [@media(hover:none)]:hidden",
        isSelected
          ? "bg-white/15 text-white"
          : "bg-white text-zinc-500 ring-1 ring-zinc-200",
      )}
    >
      {getOptionLetter(index)}
    </span>
  );
}
