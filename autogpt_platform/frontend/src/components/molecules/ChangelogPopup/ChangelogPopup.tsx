"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowSquareOut,
  ArrowRight,
  RocketLaunch,
  Sparkle,
  Wrench,
  X,
} from "@phosphor-icons/react";
import { useChangelog } from "./useChangelog";
import { ChangelogEntry } from "./changelog-data";

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
    setShowFullChangelog,
  } = useChangelog();

  if (!isVisible || !latestEntry) {
    return null;
  }

  if (showFullChangelog) {
    return (
      <ChangelogFullView
        entries={allEntries}
        onClose={() => {
          setShowFullChangelog(false);
          dismiss();
        }}
      />
    );
  }

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 w-[420px] max-w-[calc(100vw-2rem)] transition-all duration-500 ease-out ${
        isFading
          ? "pointer-events-none translate-y-2 opacity-0"
          : "translate-y-0 opacity-100"
      }`}
      onMouseEnter={pauseAutoDismiss}
      onMouseLeave={resumeAutoDismiss}
      role="dialog"
      aria-label="What's new"
    >
      <div className="overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-2xl shadow-black/10">
        {/* Header gradient bar */}
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
              <span className="rounded-full bg-white/20 px-2 py-0.5 text-xs font-medium text-white">
                {latestEntry.version}
              </span>
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

        {/* Content */}
        <div className="max-h-[60vh] overflow-y-auto">
          {/* Title & Date */}
          <div className="border-b border-neutral-100 px-5 py-3">
            <Text
              variant="body-medium"
              className="text-[15px] font-semibold leading-tight text-neutral-900"
            >
              {latestEntry.title}
            </Text>
            <Text variant="body" className="mt-1 text-xs text-neutral-500">
              {latestEntry.date}
            </Text>
          </div>

          {/* Highlights */}
          <div className="space-y-0 divide-y divide-neutral-100">
            {latestEntry.highlights.map((highlight, i) => (
              <div key={i} className="px-5 py-3">
                <div className="flex items-start gap-2.5">
                  <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-violet-100">
                    <RocketLaunch
                      className="h-3 w-3 text-violet-600"
                      weight="fill"
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <Text
                      variant="body-medium"
                      className="text-[13px] font-semibold leading-snug text-neutral-800"
                    >
                      {highlight.title}
                    </Text>
                    <Text
                      variant="body"
                      className="mt-0.5 text-xs leading-relaxed text-neutral-500"
                    >
                      {highlight.description}
                    </Text>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Improvements summary (collapsed) */}
          {latestEntry.improvements && latestEntry.improvements.length > 0 && (
            <div className="border-t border-neutral-100 px-5 py-3">
              <div className="flex items-center gap-2">
                <Wrench className="h-3.5 w-3.5 text-emerald-600" />
                <Text
                  variant="body"
                  className="text-xs font-medium text-neutral-600"
                >
                  {latestEntry.improvements.length} improvements
                </Text>
              </div>
            </div>
          )}

          {/* Fixes summary (collapsed) */}
          {latestEntry.fixes && latestEntry.fixes.length > 0 && (
            <div className="border-t border-neutral-100 px-5 py-2.5">
              <div className="flex items-center gap-2">
                <Wrench className="h-3.5 w-3.5 text-blue-600" />
                <Text
                  variant="body"
                  className="text-xs font-medium text-neutral-600"
                >
                  {latestEntry.fixes.length} bug fixes
                </Text>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-neutral-200 bg-neutral-50/80 px-5 py-2.5">
          <button
            onClick={() => setShowFullChangelog(true)}
            className="flex items-center gap-1 text-xs font-medium text-neutral-500 transition-colors hover:text-neutral-700"
          >
            View all updates
            <ArrowRight className="h-3 w-3" />
          </button>
          <a
            href={latestEntry.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs font-medium text-violet-600 transition-colors hover:text-violet-700"
          >
            Full details
            <ArrowSquareOut className="h-3 w-3" />
          </a>
        </div>
      </div>
    </div>
  );
}

/** Full changelog modal showing all entries */
function ChangelogFullView({
  entries,
  onClose,
}: {
  entries: ChangelogEntry[];
  onClose: () => void;
}) {
  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-x-0 bottom-0 top-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-8">
        <div className="relative flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-6 py-4">
            <div className="flex items-center gap-2.5">
              <Sparkle className="h-5 w-5 text-white" weight="fill" />
              <Text
                variant="body-medium"
                as="span"
                className="text-lg font-bold text-white"
              >
                Changelog
              </Text>
            </div>
            <button
              onClick={onClose}
              className="rounded-lg p-1 text-white/70 transition-colors hover:bg-white/10 hover:text-white"
              aria-label="Close changelog"
            >
              <X className="h-5 w-5" weight="bold" />
            </button>
          </div>

          {/* Entries list */}
          <div className="flex-1 overflow-y-auto">
            {entries.map((entry, entryIndex) => (
              <div
                key={entry.id}
                className={`${entryIndex > 0 ? "border-t border-neutral-200" : ""}`}
              >
                {/* Entry header */}
                <div className="sticky top-0 z-10 border-b border-neutral-100 bg-white/95 px-6 py-3 backdrop-blur-sm">
                  <div className="flex items-center justify-between">
                    <div>
                      <Text
                        variant="body-medium"
                        className="text-[15px] font-bold text-neutral-900"
                      >
                        {entry.title}
                      </Text>
                      <div className="mt-0.5 flex items-center gap-2">
                        <span className="rounded bg-violet-100 px-1.5 py-0.5 text-[11px] font-semibold text-violet-700">
                          {entry.version}
                        </span>
                        <Text
                          variant="body"
                          className="text-xs text-neutral-500"
                        >
                          {entry.date}
                        </Text>
                      </div>
                    </div>
                    <a
                      href={entry.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 rounded-md p-1.5 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-600"
                      aria-label="Open in docs"
                    >
                      <ArrowSquareOut className="h-4 w-4" />
                    </a>
                  </div>
                </div>

                {/* Highlights */}
                <div className="divide-y divide-neutral-50 px-6">
                  {entry.highlights.map((highlight, i) => (
                    <div key={i} className="py-3">
                      <div className="flex items-start gap-2.5">
                        <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-violet-100">
                          <RocketLaunch
                            className="h-3 w-3 text-violet-600"
                            weight="fill"
                          />
                        </div>
                        <div>
                          <Text
                            variant="body-medium"
                            className="text-[13px] font-semibold text-neutral-800"
                          >
                            {highlight.title}
                          </Text>
                          <Text
                            variant="body"
                            className="mt-0.5 text-xs leading-relaxed text-neutral-500"
                          >
                            {highlight.description}
                          </Text>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Improvements */}
                {entry.improvements && entry.improvements.length > 0 && (
                  <div className="border-t border-neutral-100 px-6 py-3">
                    <div className="mb-2 flex items-center gap-1.5">
                      <Wrench className="h-3.5 w-3.5 text-emerald-600" />
                      <Text
                        variant="body-medium"
                        className="text-xs font-semibold text-neutral-700"
                      >
                        Improvements
                      </Text>
                    </div>
                    <ul className="space-y-1.5">
                      {entry.improvements.map((item, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 text-xs text-neutral-600"
                        >
                          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-emerald-500" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Fixes */}
                {entry.fixes && entry.fixes.length > 0 && (
                  <div className="border-t border-neutral-100 px-6 py-3">
                    <div className="mb-2 flex items-center gap-1.5">
                      <Wrench className="h-3.5 w-3.5 text-blue-600" />
                      <Text
                        variant="body-medium"
                        className="text-xs font-semibold text-neutral-700"
                      >
                        Bug Fixes
                      </Text>
                    </div>
                    <ul className="space-y-1.5">
                      {entry.fixes.map((item, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 text-xs text-neutral-600"
                        >
                          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-blue-500" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between border-t border-neutral-200 bg-neutral-50 px-6 py-3">
            <Text variant="body" className="text-xs text-neutral-500">
              Full changelog available on the docs
            </Text>
            <a
              href="https://agpt.co/docs/platform/changelog/changelog"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-full bg-zinc-800 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-900"
            >
              View All on Docs
              <ArrowSquareOut className="h-3.5 w-3.5" />
            </a>
          </div>
        </div>
      </div>
    </>
  );
}
