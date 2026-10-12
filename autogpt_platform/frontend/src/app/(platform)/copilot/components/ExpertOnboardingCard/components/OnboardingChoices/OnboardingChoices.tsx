"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { getOptionLetter } from "../../helpers";
import { useOnboardingChoices } from "./useOnboardingChoices";

interface Props {
  options: string[];
  value: string;
  labelId: string;
  autoFocus: boolean;
  onChoose: (option: string) => void;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

/** One question's answers as lettered rows: a tap or its letter key picks
 *  the row, and the last row, "Other", opens a text box. Mounted per step,
 *  so the text box toggle starts fresh on every question. */
export function OnboardingChoices({
  options,
  value,
  labelId,
  autoFocus,
  onChoose,
  onChange,
  onSubmit,
}: Props) {
  const {
    active,
    didOpenOther,
    isTyping,
    optionRefs,
    choose,
    startTyping,
    handleLetterKey,
    handleOptionKeyDown,
    handleTextKeyDown,
  } = useOnboardingChoices({ options, value, onChoose, onChange, onSubmit });

  return (
    <div className="flex flex-col gap-2">
      {options.length > 0 && (
        <div
          role="radiogroup"
          aria-labelledby={labelId}
          aria-required="true"
          className="flex flex-col gap-2"
        >
          {options.map((option, index) => {
            const isSelected = option === value.trim();
            return (
              <button
                key={option}
                ref={(element) => {
                  optionRefs.current[index] = element;
                }}
                type="button"
                role="radio"
                aria-checked={isSelected}
                tabIndex={index === active ? 0 : -1}
                autoFocus={autoFocus && !isTyping && index === active}
                onClick={() => choose(option)}
                onKeyDown={(event) => handleOptionKeyDown(event, index)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-[15px] leading-snug transition-colors",
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
          required
          rows={2}
          aria-labelledby={labelId}
          autoFocus={autoFocus || didOpenOther}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleTextKeyDown}
          placeholder="Type your answer"
          className="resize-none rounded-xl bg-zinc-50 px-4 py-3 text-[15px] leading-relaxed text-zinc-800 ring-1 ring-zinc-200 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-400"
        />
      ) : (
        <button
          type="button"
          onClick={startTyping}
          onKeyDown={handleLetterKey}
          className="flex w-full items-center gap-3 rounded-xl border border-dashed border-zinc-200 px-3 py-2.5 text-left text-[15px] text-zinc-500 transition-colors hover:bg-zinc-50 hover:text-zinc-700"
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

function LetterBadge({ index, isSelected }: LetterBadgeProps) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-6 shrink-0 items-center justify-center rounded-md text-xs font-medium",
        isSelected
          ? "bg-white/15 text-white"
          : "bg-white text-zinc-500 ring-1 ring-zinc-200",
      )}
    >
      {getOptionLetter(index)}
    </span>
  );
}
