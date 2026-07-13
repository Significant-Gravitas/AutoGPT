"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ArrowSquareOut, Sparkle, X } from "@phosphor-icons/react";
import { useChangelog } from "./useChangelog";

export function ChangelogPopup() {
  const {
    isVisible,
    isFading,
    latestEntry,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
  } = useChangelog();

  if (!isVisible || !latestEntry) return null;

  const highlights = latestEntry.highlights
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 3);

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 w-[360px] max-w-[calc(100vw-2rem)] transition-all duration-500 ease-out ${
        isFading
          ? "pointer-events-none translate-y-2 opacity-0"
          : "translate-y-0 opacity-100"
      }`}
      onMouseEnter={pauseAutoDismiss}
      onMouseLeave={resumeAutoDismiss}
      onFocus={pauseAutoDismiss}
      onBlur={resumeAutoDismiss}
      role="dialog"
      aria-label="What's new"
    >
      <div className="overflow-hidden rounded-xl border border-border bg-background shadow-2xl shadow-black/10">
        <div className="flex items-center justify-between bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-5 py-3">
          <div className="flex items-center gap-2">
            <Sparkle className="h-4 w-4 text-white/90" weight="fill" />
            <Text
              variant="body-medium"
              as="span"
              className="text-sm font-semibold text-white"
            >
              What&apos;s New
            </Text>
          </div>
          <button
            onClick={dismiss}
            className="rounded-md p-0.5 text-white/70 transition-colors hover:bg-white/10 hover:text-white"
            aria-label="Dismiss changelog"
          >
            <X className="h-4 w-4" weight="bold" />
          </button>
        </div>

        <div className="px-5 py-4">
          <Text variant="body" className="text-xs text-muted-foreground">
            {latestEntry.dateRange}
          </Text>
          <ul className="mt-2 space-y-1">
            {highlights.map((item) => (
              <li key={item} className="flex gap-2">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent" />
                <Text
                  variant="body"
                  as="span"
                  className="text-sm leading-snug text-foreground"
                >
                  {item}
                </Text>
              </li>
            ))}
          </ul>
        </div>

        <div className="flex justify-end border-t border-border bg-secondary/50 px-5 py-2.5">
          <a
            href={latestEntry.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs font-medium text-accent transition-colors hover:text-accent/80"
          >
            View changelog
            <ArrowSquareOut className="h-3 w-3" />
          </a>
        </div>
      </div>
    </div>
  );
}
