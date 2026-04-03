"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowRight,
  ArrowSquareOut,
  Sparkle,
  X,
} from "@phosphor-icons/react";
import { CHANGELOG_INDEX_URL } from "./changelog-constants";
import { ChangelogEntry, useChangelog } from "./useChangelog";

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

  if (!isVisible || !latestEntry) return null;

  if (showFullChangelog) {
    return (
      <ChangelogEmbed
        entries={allEntries}
        initialUrl={latestEntry.url}
        onClose={() => {
          setShowFullChangelog(false);
          dismiss();
        }}
      />
    );
  }

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
      <div className="overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-2xl shadow-black/10">
        {/* Header */}
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

        {/* Latest entry */}
        <div className="px-5 py-4">
          <Text
            variant="body-medium"
            className="text-[15px] font-semibold leading-snug text-neutral-900"
          >
            {latestEntry.title}
          </Text>
          <Text variant="body" className="mt-1 text-xs text-neutral-500">
            {latestEntry.date}
          </Text>
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between border-t border-neutral-200 bg-neutral-50/80 px-5 py-2.5">
          <button
            onClick={() => setShowFullChangelog(true)}
            className="flex items-center gap-1 text-xs font-medium text-neutral-500 transition-colors hover:text-neutral-700"
          >
            Read more
            <ArrowRight className="h-3 w-3" />
          </button>
          <a
            href={latestEntry.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs font-medium text-violet-600 transition-colors hover:text-violet-700"
          >
            Open in docs
            <ArrowSquareOut className="h-3 w-3" />
          </a>
        </div>
      </div>
    </div>
  );
}

/** Full changelog view — sidebar with entry list + iframe embed of the selected entry */
function ChangelogEmbed({
  entries,
  initialUrl,
  onClose,
}: {
  entries: ChangelogEntry[];
  initialUrl: string;
  onClose: () => void;
}) {
  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-4 z-50 flex overflow-hidden rounded-2xl border border-neutral-200 bg-white shadow-2xl sm:inset-6 md:inset-10">
        {/* Sidebar — entry list */}
        <div className="hidden w-72 shrink-0 flex-col border-r border-neutral-200 bg-neutral-50 md:flex">
          {/* Sidebar header */}
          <div className="flex items-center justify-between bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-4 py-3">
            <div className="flex items-center gap-2">
              <Sparkle className="h-4 w-4 text-white" weight="fill" />
              <Text
                variant="body-medium"
                as="span"
                className="text-sm font-bold text-white"
              >
                Changelog
              </Text>
            </div>
          </div>

          {/* Entry list */}
          <nav className="flex-1 overflow-y-auto p-2">
            {entries.map((entry) => (
              <a
                key={entry.slug}
                href={entry.url}
                target="changelog-embed"
                className="mb-1 block rounded-lg px-3 py-2.5 transition-colors hover:bg-neutral-100"
              >
                <Text
                  variant="body-medium"
                  className="text-[13px] font-medium leading-snug text-neutral-800"
                >
                  {entry.title}
                </Text>
                <Text
                  variant="body"
                  className="mt-0.5 text-[11px] text-neutral-500"
                >
                  {entry.date}
                </Text>
              </a>
            ))}
          </nav>

          {/* Sidebar footer */}
          <div className="border-t border-neutral-200 p-3">
            <a
              href={CHANGELOG_INDEX_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-1.5 rounded-full bg-zinc-800 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-zinc-900"
            >
              View on Docs
              <ArrowSquareOut className="h-3 w-3" />
            </a>
          </div>
        </div>

        {/* Main content — iframe embed */}
        <div className="flex flex-1 flex-col">
          {/* Top bar (mobile: shows title + close; desktop: just close) */}
          <div className="flex items-center justify-between border-b border-neutral-200 bg-white px-4 py-2">
            <div className="flex items-center gap-2 md:hidden">
              <Sparkle
                className="h-4 w-4 text-violet-600"
                weight="fill"
              />
              <Text
                variant="body-medium"
                as="span"
                className="text-sm font-semibold text-neutral-800"
              >
                Changelog
              </Text>
            </div>
            <a
              href={CHANGELOG_INDEX_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="hidden items-center gap-1 text-xs text-neutral-500 transition-colors hover:text-neutral-700 md:flex"
            >
              Open in new tab
              <ArrowSquareOut className="h-3 w-3" />
            </a>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-600"
              aria-label="Close changelog"
            >
              <X className="h-5 w-5" weight="bold" />
            </button>
          </div>

          {/* Iframe */}
          <iframe
            name="changelog-embed"
            src={initialUrl}
            className="flex-1 border-0"
            title="AutoGPT Platform Changelog"
            sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox"
          />
        </div>
      </div>
    </>
  );
}
