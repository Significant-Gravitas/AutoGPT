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

// Shortcut cues only help where there is a keyboard; on touch screens they
// are noise, so they show only where the pointer can hover.
const KEY_HINT =
  "hidden rounded border border-zinc-200 px-1 text-[11px] font-medium leading-4 text-zinc-400 [@media(hover:hover)]:inline";

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
    <div className="flex items-center justify-between gap-2 px-4 pb-4 pt-4 sm:px-5">
      <button
        type="button"
        aria-label="Previous question"
        aria-keyshortcuts="Backspace"
        disabled={!canGoBack}
        onClick={onBack}
        className="flex size-8 items-center justify-center rounded-md text-zinc-400 outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400 enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:invisible [@media(hover:none)]:size-11"
      >
        <Icon icon={ArrowLeft01Icon} size={16} />
      </button>
      <div className="flex items-center gap-2">
        <button
          type="button"
          aria-keyshortcuts="Escape"
          onClick={onSkip}
          disabled={isSending}
          className="flex h-8 items-center gap-2 rounded-md border border-zinc-200 bg-white px-3.5 text-sm font-medium text-zinc-700 outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400 enabled:hover:border-zinc-300 enabled:hover:bg-zinc-50 disabled:opacity-50 [@media(hover:none)]:h-11"
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
            "flex h-8 items-center gap-1.5 rounded-md px-4 text-sm font-medium outline-none transition-all duration-200 focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 enabled:active:scale-95 [@media(hover:none)]:h-11",
            isEnabled
              ? "bg-zinc-900 text-white hover:bg-zinc-800"
              : "bg-zinc-100 text-zinc-400",
          )}
        >
          {label}
          <Icon icon={isReview ? SentIcon : ArrowRight01Icon} size={14} />
          {isEnabled && (
            <kbd
              aria-hidden="true"
              className={cn(KEY_HINT, "border-white/25 text-white/70")}
            >
              Enter ↵
            </kbd>
          )}
        </button>
      </div>
    </div>
  );
}
