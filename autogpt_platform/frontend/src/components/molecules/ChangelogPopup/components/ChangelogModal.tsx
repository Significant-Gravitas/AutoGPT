"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ArrowSquareOut, Sparkle, X } from "@phosphor-icons/react";
import dynamic from "next/dynamic";
import { CHANGELOG_BASE_URL } from "../changelog-constants";
import { ChangelogEntry } from "../useChangelog";

const ChangelogMarkdownContent = dynamic(
  () =>
    import("./ChangelogMarkdownContent").then(
      (mod) => mod.ChangelogMarkdownContent,
    ),
  {
    loading: () => (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-border border-t-accent" />
        Loading…
      </div>
    ),
  },
);

interface ChangelogModalProps {
  entries: ChangelogEntry[];
  selectedEntry: ChangelogEntry | null;
  entryMarkdown: string | null;
  isLoadingMarkdown: boolean;
  onSelectEntry: (entry: ChangelogEntry) => void;
  onClose: () => void;
}

export function ChangelogModal({
  entries,
  selectedEntry,
  entryMarkdown,
  isLoadingMarkdown,
  onSelectEntry,
  onClose,
}: ChangelogModalProps) {
  return (
    <>
      <div
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="fixed inset-4 z-50 flex overflow-hidden rounded-2xl border border-border bg-background shadow-2xl sm:inset-6 md:inset-10">
        <div className="hidden w-72 shrink-0 flex-col border-r border-border bg-secondary md:flex">
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

          <nav className="flex-1 overflow-y-auto p-2">
            {entries.map((entry) => (
              <button
                key={entry.slug}
                onClick={() => onSelectEntry(entry)}
                className={`mb-1 block w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
                  selectedEntry?.slug === entry.slug
                    ? "bg-accent/10 ring-1 ring-accent/20"
                    : "hover:bg-secondary"
                }`}
              >
                <Text
                  variant="body-medium"
                  className="text-[13px] font-medium leading-snug text-foreground"
                >
                  {entry.highlights}
                </Text>
                <Text
                  variant="body"
                  className="mt-0.5 text-[11px] text-muted-foreground"
                >
                  {entry.dateRange}
                </Text>
              </button>
            ))}
          </nav>

          <div className="border-t border-border p-3">
            <a
              href={CHANGELOG_BASE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-1.5 rounded-full bg-primary px-3 py-2 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              View on Docs
              <ArrowSquareOut className="h-3 w-3" />
            </a>
          </div>
        </div>

        <div className="flex flex-1 flex-col">
          <div className="flex items-center justify-between border-b border-border bg-background px-4 py-2">
            <div className="flex items-center gap-2 md:hidden">
              <Sparkle className="h-4 w-4 text-accent" weight="fill" />
              {entries.length > 0 && (
                <select
                  className="max-w-[200px] truncate rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground"
                  value={selectedEntry?.slug || ""}
                  onChange={(e) => {
                    const entry = entries.find((en) => en.slug === e.target.value);
                    if (entry) onSelectEntry(entry);
                  }}
                >
                  {entries.map((entry) => (
                    <option key={entry.slug} value={entry.slug}>
                      {entry.dateRange}
                    </option>
                  ))}
                </select>
              )}
            </div>
            {selectedEntry && (
              <a
                href={selectedEntry.url}
                target="_blank"
                rel="noopener noreferrer"
                className="hidden items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground md:flex"
              >
                Open in docs
                <ArrowSquareOut className="h-3 w-3" />
              </a>
            )}
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              aria-label="Close changelog"
            >
              <X className="h-5 w-5" weight="bold" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-6 md:px-10">
            {isLoadingMarkdown && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-border border-t-accent" />
                Loading…
              </div>
            )}
            {entryMarkdown && (
              <ChangelogMarkdownContent markdown={entryMarkdown} />
            )}
            {!isLoadingMarkdown && !entryMarkdown && selectedEntry && (
              <div className="text-sm text-muted-foreground">
                Could not load changelog entry.{" "}
                <a
                  href={selectedEntry.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent underline"
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
