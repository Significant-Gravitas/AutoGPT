"use client";

import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { ResolvedWorkflowPage } from "@/app/api/__generated__/models/resolvedWorkflowPage";
import { Text } from "@/components/atoms/Text/Text";
import { OutdatedNotice } from "./components/OutdatedNotice";
import { PageSectionView } from "./components/PageSectionView";

interface Props {
  page: ResolvedWorkflowPage;
  /** Given in the library, where the form runs; absent, the page is a preview. */
  agent?: LibraryAgent;
  headerSlot?: React.ReactNode;
  onRunStarted?: (run: GraphExecutionMeta) => void;
  onSelectRun?: (runID: string) => void;
  onRemoveSection?: (index: number) => void;
}

export function WorkflowPage({
  page,
  agent,
  headerSlot,
  onRunStarted,
  onSelectRun,
  onRemoveSection,
}: Props) {
  return (
    <article
      aria-label={page.title}
      className="flex w-full min-w-0 flex-col gap-4"
      data-testid="workflow-page"
    >
      <header className="flex items-start justify-between gap-3">
        <Text variant="h3" as="h2" className="break-words">
          {page.title}
        </Text>
        {headerSlot}
      </header>
      {page.left_out?.length ? (
        <OutdatedNotice
          workflowName={agent?.name ?? page.title}
          count={page.left_out.length}
          canAsk={Boolean(agent)}
        />
      ) : null}
      {page.sections.map((section) => (
        <PageSectionView
          key={section.index}
          section={section}
          agent={agent}
          onRunStarted={onRunStarted}
          onSelectRun={onSelectRun}
          onRemove={onRemoveSection}
        />
      ))}
    </article>
  );
}
