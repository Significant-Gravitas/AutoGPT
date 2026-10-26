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
                  // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
                  "flex w-full items-center gap-3 rounded-md px-3 py-3 text-left text-base leading-snug outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 sm:py-2.5 sm:text-body-md sm:leading-snug [@media(hover:none)]:min-h-11",
                  isSelected
                    ? "bg-action-hover text-action-foreground"
                    : "bg-surface-muted text-foreground-emphasis ring-1 ring-separator hover:bg-surface-hover",
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
          // TODO(tokens): no fitting role for focus:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
          className="resize-none rounded-md bg-surface-muted px-4 py-3 text-base leading-relaxed text-foreground-emphasis ring-1 ring-border-base transition-shadow placeholder:text-foreground-subtle focus:outline-none focus:ring-zinc-400 sm:text-body-md sm:leading-relaxed"
        />
      ) : (
        <button
          type="button"
          onClick={startTyping}
          // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
          className="flex w-full items-center gap-3 rounded-md border border-dashed border-border-base px-3 py-3 text-left text-base text-foreground-tertiary outline-none transition-colors hover:bg-surface-muted hover:text-foreground-body focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 sm:py-2.5 sm:text-body-md [@media(hover:none)]:min-h-11"
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
          ? "bg-surface/15 text-action-foreground"
          : "bg-surface text-foreground-tertiary ring-1 ring-border-base",
      )}
    >
      {getOptionLetter(index)}
    </span>
  );
}
