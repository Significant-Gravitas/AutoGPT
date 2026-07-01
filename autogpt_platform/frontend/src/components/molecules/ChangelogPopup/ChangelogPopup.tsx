"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ArrowRight, ArrowSquareOut, Sparkle, X } from "@phosphor-icons/react";
import { ChangelogModal } from "./components/ChangelogModal";
import { useChangelog } from "./useChangelog";

export function ChangelogPopup() {
  const {
    isVisible,
    latestEntry,
    allEntries,
    isFading,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
    showFullChangelog,
    openFullChangelog,
    closeFullChangelog,
    selectedEntry,
    selectEntry,
    entryMarkdown,
    isLoadingMarkdown,
  } = useChangelog();

  if (showFullChangelog) {
    return (
      <ChangelogModal
        entries={allEntries}
        selectedEntry={selectedEntry}
        entryMarkdown={entryMarkdown}
        isLoadingMarkdown={isLoadingMarkdown}
        onSelectEntry={selectEntry}
        onClose={closeFullChangelog}
      />
    );
  }

  if (!isVisible || !latestEntry) return null;

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 w-[400px] max-w-[calc(100vw-2rem)] transition-all duration-500 ease-out ${
        isFading
          ? "pointer-events-none translate-y-2 opacity-0"
          : "translate-y-0 opacity-100"
      }`}
      onMouseEnter={pauseAutoDismiss}
      onMouseLeave={resumeAutoDismiss}
      role="dialog"
      aria-label="What's new"
    >
      <div className="overflow-hidden rounded-xl border border-border bg-background shadow-2xl shadow-black/10">
        <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-5 py-3">
          <div className="flex items-center justify-between">
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
        </div>

        <div className="px-5 py-4">
          <Text
            variant="body-medium"
            className="text-[15px] font-semibold leading-snug text-foreground"
          >
            {latestEntry.highlights}
          </Text>
          <Text variant="body" className="mt-1 text-xs text-muted-foreground">
            {latestEntry.dateRange}
          </Text>
        </div>

        <div className="flex items-center justify-between border-t border-border bg-secondary/50 px-5 py-2.5">
          <button
            onClick={() => openFullChangelog(latestEntry)}
            className="flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Read more
            <ArrowRight className="h-3 w-3" />
          </button>
          <a
            href={latestEntry.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs font-medium text-accent transition-colors hover:text-accent/80"
          >
            Open in docs
            <ArrowSquareOut className="h-3 w-3" />
          </a>
        </div>
      </div>
    </div>
  );
}
