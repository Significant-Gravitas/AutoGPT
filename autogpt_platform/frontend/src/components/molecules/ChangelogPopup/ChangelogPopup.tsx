"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowRight,
  ArrowSquareOut,
  Sparkle,
  X,
} from "@phosphor-icons/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CHANGELOG_BASE_URL } from "./changelog-constants";
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

        {/* Latest entry summary */}
        <div className="px-5 py-4">
          <Text
            variant="body-medium"
            className="text-[15px] font-semibold leading-snug text-neutral-900"
          >
            {latestEntry.highlights}
          </Text>
          <Text variant="body" className="mt-1 text-xs text-neutral-500">
            {latestEntry.dateRange}
          </Text>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-neutral-200 bg-neutral-50/80 px-5 py-2.5">
          <button
            onClick={() => openFullChangelog(latestEntry)}
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

function ChangelogModal({
  entries,
  selectedEntry,
  entryMarkdown,
  isLoadingMarkdown,
  onSelectEntry,
  onClose,
}: {
  entries: ChangelogEntry[];
  selectedEntry: ChangelogEntry | null;
  entryMarkdown: string | null;
  isLoadingMarkdown: boolean;
  onSelectEntry: (entry: ChangelogEntry) => void;
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
        {/* Sidebar */}
        <div className="hidden w-72 shrink-0 flex-col border-r border-neutral-200 bg-neutral-50 md:flex">
          {/* Sidebar header */}
          <div className="flex items-center gap-2 bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-4 py-3">
            <Sparkle className="h-4 w-4 text-white" weight="fill" />
            <Text
              variant="body-medium"
              as="span"
              className="text-sm font-bold text-white"
            >
              Changelog
            </Text>
          </div>

          {/* Entry list */}
          <nav className="flex-1 overflow-y-auto p-2">
            {entries.map((entry) => (
              <button
                key={entry.slug}
                onClick={() => onSelectEntry(entry)}
                className={`mb-1 block w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
                  selectedEntry?.slug === entry.slug
                    ? "bg-violet-100 ring-1 ring-violet-200"
                    : "hover:bg-neutral-100"
                }`}
              >
                <Text
                  variant="body-medium"
                  className="text-[13px] font-medium leading-snug text-neutral-800"
                >
                  {entry.highlights}
                </Text>
                <Text
                  variant="body"
                  className="mt-0.5 text-[11px] text-neutral-500"
                >
                  {entry.dateRange}
                </Text>
              </button>
            ))}
          </nav>

          {/* Sidebar footer */}
          <div className="border-t border-neutral-200 p-3">
            <a
              href={CHANGELOG_BASE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-1.5 rounded-full bg-zinc-800 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-zinc-900"
            >
              View on Docs
              <ArrowSquareOut className="h-3 w-3" />
            </a>
          </div>
        </div>

        {/* Main content */}
        <div className="flex flex-1 flex-col">
          {/* Top bar */}
          <div className="flex items-center justify-between border-b border-neutral-200 bg-white px-4 py-2">
            <div className="flex items-center gap-2 md:hidden">
              <Sparkle className="h-4 w-4 text-violet-600" weight="fill" />
              <Text
                variant="body-medium"
                as="span"
                className="text-sm font-semibold text-neutral-800"
              >
                Changelog
              </Text>
            </div>
            {selectedEntry && (
              <a
                href={selectedEntry.url}
                target="_blank"
                rel="noopener noreferrer"
                className="hidden items-center gap-1 text-xs text-neutral-500 transition-colors hover:text-neutral-700 md:flex"
              >
                Open in docs
                <ArrowSquareOut className="h-3 w-3" />
              </a>
            )}
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-600"
              aria-label="Close changelog"
            >
              <X className="h-5 w-5" weight="bold" />
            </button>
          </div>

          {/* Markdown content */}
          <div className="flex-1 overflow-y-auto px-6 py-6 md:px-10">
            {isLoadingMarkdown && (
              <div className="flex items-center gap-2 text-sm text-neutral-500">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-neutral-300 border-t-violet-600" />
                Loading…
              </div>
            )}
            {entryMarkdown && (
              <ReactMarkdown
                className="prose prose-sm max-w-none prose-headings:text-neutral-900 prose-p:text-neutral-600 prose-a:text-violet-600 prose-a:no-underline hover:prose-a:underline prose-strong:text-neutral-800 prose-img:rounded-lg prose-img:shadow-md"
                remarkPlugins={[remarkGfm]}
                components={{
                  // Open all links in new tab
                  a: ({ children, href, ...props }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      {...props}
                    >
                      {children}
                    </a>
                  ),
                  // Render images with proper styling
                  // eslint-disable-next-line @next/next/no-img-element
                  img: ({ src, alt, ...props }) => (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={src}
                      alt={alt || ""}
                      className="my-4 h-auto max-w-full rounded-lg shadow-md"
                      loading="lazy"
                      {...props}
                    />
                  ),
                }}
              >
                {entryMarkdown}
              </ReactMarkdown>
            )}
            {!isLoadingMarkdown && !entryMarkdown && selectedEntry && (
              <div className="text-sm text-neutral-500">
                Could not load changelog entry.{" "}
                <a
                  href={selectedEntry.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-violet-600 underline"
                >
                  View on docs
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
