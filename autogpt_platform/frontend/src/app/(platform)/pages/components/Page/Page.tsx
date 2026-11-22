"use client";

import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { PageSectionView } from "./components/PageSectionView";

export interface PageLive {
  /** Keys the runs this page follows; a stored page's id, or a pinned workflow's. */
  pageKey: string;
  onRunStarted(libraryAgentID: string, runID: string): void;
  onReviewed(): void;
}

export interface PageEditing {
  onMove(index: number, by: -1 | 1): void;
  onRemove(index: number): void;
  onChangeWithOtto(index: number, title: string | null): void;
}

interface Props {
  page: ResolvedPage;
  /** Absent in a chat preview, where nothing runs. */
  live?: PageLive;
  editing?: PageEditing;
  headerSlot?: ReactNode;
}

export function Page({ page, live, editing, headerSlot }: Props) {
  const last = page.sections.length - 1;
  return (
    <article
      aria-label={page.title}
      className="flex w-full min-w-0 flex-col gap-4"
      data-testid="page"
    >
      <header className="flex items-start justify-between gap-3">
        <Text variant="h3" as="h2" className="break-words">
          {page.title}
        </Text>
        {headerSlot}
      </header>
      {page.sections.map((section, position) => (
        <PageSectionView
          key={`${section.index}-${section.kind}`}
          section={section}
          live={live}
          editing={editing}
          isFirst={position === 0}
          isLast={position === last}
        />
      ))}
    </article>
  );
}
