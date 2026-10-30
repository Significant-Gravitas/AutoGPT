"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  SentIcon,
} from "@hugeicons/core-free-icons";

interface Props {
  canGoBack: boolean;
  isReady: boolean;
  isReview: boolean;
  isNextReview: boolean;
  isSending: boolean;
  onBack: () => void;
  onSkip: () => void;
  onNext: () => void;
}

// Keyboard cues show only where the pointer can hover and the footer fits them
// with each button on one line, which the Team panel's card does not.
const KEY_HINT =
  "hidden rounded border border-border-base px-1 text-caption-sm font-medium leading-4 text-foreground-subtle [@media(hover:hover)]:[@container(min-width:24.5rem)]:inline";

export function OnboardingFooter({
  canGoBack,
  isReady,
  isReview,
  isNextReview,
  isSending,
  onBack,
  onSkip,
  onNext,
}: Props) {
  const isEnabled = isReady && !isSending;
  const label = isReview ? "Send" : isNextReview ? "Review" : "Next";
  const ariaLabel = isReview
    ? "Send answers"
    : isNextReview
      ? "Review answers"
      : "Next question";

  return (
    <div className="[container-type:inline-size] flex items-center justify-between gap-2 px-4 pt-4 pb-4 sm:px-5">
      <button
        type="button"
        aria-label="Previous question"
        aria-keyshortcuts="Backspace"
        disabled={!canGoBack}
        onClick={onBack}
        className="flex size-8 items-center justify-center rounded-md text-foreground-subtle transition-colors outline-none focus-visible:ring-2 focus-visible:ring-foreground-subtle enabled:hover:bg-surface-hover enabled:hover:text-foreground-secondary disabled:invisible [@media(hover:none)]:size-11"
      >
        <Icon icon={ArrowLeft01Icon} size={16} />
      </button>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <button
          type="button"
          aria-keyshortcuts="Escape"
          onClick={onSkip}
          disabled={isSending}
          className="flex h-8 items-center gap-2 rounded-md border border-border-base bg-surface px-3.5 text-sm font-medium whitespace-nowrap text-foreground-body transition-colors outline-none focus-visible:ring-2 focus-visible:ring-foreground-subtle enabled:hover:border-border-strong enabled:hover:bg-surface-muted disabled:opacity-50 [@media(hover:none)]:h-11"
        >
          Skip questions
          <kbd aria-hidden="true" className={KEY_HINT}>
            Esc
          </kbd>
        </button>
        <button
          // Remounted on the review step so autoFocus lands on Send: the
          // question the focus was in has just unmounted.
          key={isReview ? "send" : "next"}
          type="button"
          aria-label={ariaLabel}
          aria-keyshortcuts="Enter"
          disabled={!isEnabled}
          autoFocus={isReview}
          onClick={onNext}
          className={cn(
            "flex h-8 items-center gap-1.5 rounded-md px-4 text-sm font-medium whitespace-nowrap transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-foreground-subtle focus-visible:ring-offset-1 enabled:active:scale-95 [@media(hover:none)]:h-11",
            isEnabled
              ? "bg-action-hover text-action-foreground hover:bg-action"
              : "bg-surface-hover text-foreground-subtle",
          )}
        >
          {label}
          <Icon icon={isReview ? SentIcon : ArrowRight01Icon} size={14} />
          {isEnabled && (
            <kbd
              aria-hidden="true"
              className={cn(
                KEY_HINT,
                "border-action-foreground/25 text-action-foreground/70",
              )}
            >
              Enter ↵
            </kbd>
          )}
        </button>
      </div>
    </div>
  );
}
