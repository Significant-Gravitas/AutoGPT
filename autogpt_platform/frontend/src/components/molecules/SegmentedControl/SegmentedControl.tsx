"use client";

import type { IconSvgElement } from "@hugeicons/react";
import { LayoutGroup, motion, useReducedMotion } from "framer-motion";
import { useId, useRef, type KeyboardEvent } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  count?: number;
  icon?: IconSvgElement;
}

interface Props<T extends string> {
  value: T;
  onChange(v: T): void;
  options: SegmentedOption<T>[];
  ariaLabel: string;
  size?: "sm" | "md";
}

const PILL_TRANSITION = { duration: 0.4, ease: [0.22, 1, 0.36, 1] } as const;

function nextIndex(e: KeyboardEvent, current: number, length: number) {
  if (isKey(e, "ArrowRight", "ArrowDown")) return (current + 1) % length;
  if (isKey(e, "ArrowLeft", "ArrowUp")) return (current - 1 + length) % length;
  if (isKey(e, "Home")) return 0;
  if (isKey(e, "End")) return length - 1;
  return null;
}

/**
 * Two to six options on a light wash; a white pill glides to the picked one.
 * A radiogroup: arrow keys, Home and End move and select.
 */
export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  ariaLabel,
  size = "md",
}: Props<T>) {
  const groupId = useId();
  const reduceMotion = useReducedMotion();
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const activeIndex = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const index = nextIndex(e, activeIndex, options.length);
    if (index === null) return;
    e.preventDefault();
    onChange(options[index].value);
    buttons.current[index]?.focus();
  }

  return (
    <LayoutGroup id={groupId}>
      <div
        role="radiogroup"
        aria-label={ariaLabel}
        onKeyDown={handleKeyDown}
        className={cn(
          "inline-flex max-w-full items-center overflow-x-auto rounded-control bg-surface-wash p-0.5",
          size === "sm" ? "h-7" : "h-8",
        )}
      >
        {options.map((option, index) => {
          const active = index === activeIndex;
          return (
            <button
              key={option.value}
              ref={(el) => {
                buttons.current[index] = el;
              }}
              type="button"
              role="radio"
              aria-checked={active}
              tabIndex={active ? 0 : -1}
              onClick={() => onChange(option.value)}
              className={cn(
                "relative flex h-full shrink-0 items-center rounded-control transition-colors duration-150 ease-out-expo focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none",
                size === "sm" ? "px-2" : "px-2.5",
                !active && "hover:bg-surface-wash",
              )}
            >
              {active ? (
                <motion.span
                  layoutId="segmented-pill"
                  transition={reduceMotion ? { duration: 0 } : PILL_TRANSITION}
                  className="absolute inset-0 rounded-control bg-surface shadow-pill"
                />
              ) : null}
              <span className="relative flex items-center gap-1.5">
                {option.icon ? (
                  <Icon
                    icon={option.icon}
                    size={14}
                    tone={active ? "ink" : "quiet"}
                    motion="colors"
                  />
                ) : null}
                <Text
                  as="span"
                  variant="inbox-medium"
                  tone={active ? "ink" : "text"}
                  motion="colors"
                  className="whitespace-nowrap"
                >
                  {option.label}
                </Text>
                {option.count !== undefined ? (
                  <Text as="span" variant="small" tone="quiet" tabular>
                    <NumberTicker value={option.count} />
                  </Text>
                ) : null}
              </span>
            </button>
          );
        })}
      </div>
    </LayoutGroup>
  );
}
