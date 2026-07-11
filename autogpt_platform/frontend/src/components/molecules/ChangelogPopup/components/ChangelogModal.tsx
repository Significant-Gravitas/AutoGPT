"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ArrowSquareOut, Sparkle, X } from "@phosphor-icons/react";
import { CHANGELOG_BASE_URL } from "../changelog-constants";
import { ChangelogEntry } from "../helpers";

interface Props {
  entries: ChangelogEntry[];
  onClose: () => void;
}

export function ChangelogModal({ entries, onClose }: Props) {
  return (
    <>
      <div
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="fixed left-1/2 top-1/2 z-50 flex max-h-[80vh] w-[calc(100%-2rem)] max-w-lg -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-2xl border border-border bg-background shadow-2xl">
        <div className="flex items-center justify-between bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-5 py-3">
          <div className="flex items-center gap-2">
            <Sparkle className="h-4 w-4 text-white" weight="fill" />
            <Text
              variant="body-medium"
              as="span"
              className="text-sm font-bold text-white"
            >
              What&apos;s New
            </Text>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-0.5 text-white/80 transition-colors hover:bg-white/10 hover:text-white"
            aria-label="Close changelog"
          >
            <X className="h-5 w-5" weight="bold" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto p-2">
          {entries.map((entry) => (
            <a
              key={entry.slug}
              href={entry.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block rounded-lg px-3 py-3 transition-colors hover:bg-secondary"
            >
              <div className="flex items-start justify-between gap-2">
                <Text
                  variant="body-medium"
                  className="text-sm font-medium leading-snug text-foreground"
                >
                  {entry.highlights}
                </Text>
                <ArrowSquareOut className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              </div>
              <Text
                variant="body"
                className="mt-0.5 text-xs text-muted-foreground"
              >
                {entry.dateRange}
              </Text>
            </a>
          ))}
        </nav>

        <div className="border-t border-border p-3">
          <a
            href={CHANGELOG_BASE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-1.5 rounded-full bg-primary px-3 py-2 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            View all on docs
            <ArrowSquareOut className="h-3 w-3" />
          </a>
        </div>
      </div>
    </>
  );
}
