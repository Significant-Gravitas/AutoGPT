"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { Fragment, useEffect, useRef, type ReactNode } from "react";
import { isKey } from "@/lib/keyboard";

interface Props {
  /** Expected trimmed and deduped — `toOptions` establishes that invariant,
   *  and the option strings double as React keys and selection identities. */
  options: string[];
  value: string;
  labelId: string;
  focusActiveOption: boolean;
  /** The option Otto would pick, marked Recommended. */
  recommended?: string | null;
  onChange: (value: string) => void;
  /** Copies the option into free text, for an answer that is close but not
   *  quite right as written. */
  onEdit?: (value: string) => void;
  /** The option was chosen as the answer: a click or its number answers at
   *  once. Without it a click only selects. */
  onPick?: (value: string) => void;
  onSubmit: () => void;
  editable?: boolean;
  required?: boolean;
  /** Rendered inside the group, between the selected row and the next one. */
  selectedContent?: ReactNode;
}

/** The options for one question as numbered rows in a real radiogroup: a
 *  single stop in the tab order, arrows moving between rows. A click, Enter
 *  or the row's number answers with it at once. */
export function QuestionOptionList({
  options,
  value,
  labelId,
  focusActiveOption,
  recommended,
  onChange,
  onEdit,
  onPick,
  onSubmit,
  editable = true,
  required = true,
  selectedContent,
}: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const showPencil = editable && Boolean(onEdit);
  const selected = options.indexOf(value.trim());
  const active = selected === -1 ? 0 : selected;

  useEffect(() => {
    if (focusActiveOption) refs.current[active]?.focus();
  }, [focusActiveOption, active]);

  function pick(option: string) {
    if (onPick) onPick(option);
    else onChange(option);
  }

  function moveTo(index: number) {
    onChange(options[index]);
    refs.current[index]?.focus();
  }

  function handleKeyDown(event: React.KeyboardEvent, index: number) {
    if (isKey(event, "Enter")) {
      event.preventDefault();
      if (options[index] === value.trim()) onSubmit();
      else pick(options[index]);
      return;
    }
    if (isKey(event, " ")) {
      event.preventDefault();
      onChange(options[index]);
      return;
    }
    // A row's own number answers with it, wherever the focus is in the list.
    const digit = Number(event.key);
    if (Number.isInteger(digit) && digit >= 1 && digit <= options.length) {
      event.preventDefault();
      pick(options[digit - 1]);
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
      aria-required={required}
      className="flex flex-col"
    >
      {options.map((option, index) => {
        const isSelected = option === value.trim();
        return (
          <Fragment key={option}>
            <div className="group/option relative">
              <button
                ref={(element) => {
                  refs.current[index] = element;
                }}
                type="button"
                role="radio"
                aria-checked={isSelected}
                tabIndex={index === active ? 0 : -1}
                onClick={() => pick(option)}
                onKeyDown={(event) => handleKeyDown(event, index)}
                className={cn(
                  "flex min-h-9 w-full items-center gap-2.5 rounded-lg py-1 pl-1 text-left transition-colors duration-100 ease-out outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-inset",
                  showPencil ? "pr-10" : "pr-2",
                  isSelected ? "bg-surface-wash" : "hover:bg-surface-wash",
                )}
              >
                <span
                  aria-hidden
                  className={cn(
                    "flex size-[1.375rem] shrink-0 items-center justify-center rounded-md text-caption font-medium tabular-nums",
                    isSelected
                      ? "bg-foreground-ink text-surface"
                      : "bg-surface-wash text-foreground-quiet",
                  )}
                >
                  {index + 1}
                </span>
                <Text
                  as="span"
                  variant="inbox"
                  tone="ink"
                  className="min-w-0 flex-1 text-pretty"
                >
                  {option}
                </Text>
                {option === recommended ? (
                  <Text
                    as="span"
                    variant="inbox"
                    tone="faint"
                    className="shrink-0"
                  >
                    Recommended
                  </Text>
                ) : null}
              </button>
              {showPencil && (
                <button
                  type="button"
                  aria-label={`Edit ${option}`}
                  title="Edit before sending"
                  onClick={() => onEdit?.(option)}
                  className="absolute top-1/2 right-1.5 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-foreground-quiet opacity-0 transition-opacity duration-100 outline-none group-hover/option:opacity-100 hover:bg-surface-wash hover:text-foreground-ink focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-focus-inbox"
                >
                  <Icon icon={PencilEdit02Icon} size={13} tone="inherit" />
                </button>
              )}
            </div>
            {isSelected && selectedContent}
          </Fragment>
        );
      })}
    </div>
  );
}
