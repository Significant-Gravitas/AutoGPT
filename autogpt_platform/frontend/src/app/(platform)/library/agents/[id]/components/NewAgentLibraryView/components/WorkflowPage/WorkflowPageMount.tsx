"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import type { ReactNode } from "react";
import { useWorkflowPageMount } from "./useWorkflowPageMount";
import { WorkflowPage } from "./WorkflowPage";

interface Props {
  agent: LibraryAgent;
  onSelectRun?: (runID: string) => void;
  /** The stock view, shown when the workflow has no page. */
  children: ReactNode;
}

export function WorkflowPageMount({ agent, onSelectRun, children }: Props) {
  const mount = useWorkflowPageMount(agent);

  if (!mount.enabled) return children;
  if (mount.isLoading) {
    return <Skeleton className="h-64 w-full rounded-2xl" />;
  }
  if (!mount.page) return children;

  return (
    <div className="mx-auto w-full max-w-3xl p-4">
      <WorkflowPage
        page={mount.page}
        agent={agent}
        onRunStarted={mount.onRunStarted}
        onSelectRun={onSelectRun}
        onRemoveSection={mount.onRemoveSection}
        headerSlot={
          <div className="flex shrink-0 gap-1">
            {mount.page.has_previous ? (
              <Button variant="ghost" size="xs" onClick={mount.onRestore}>
                Undo last change
              </Button>
            ) : null}
            <Button variant="ghost" size="xs" onClick={mount.onReset}>
              Use the standard view
            </Button>
          </div>
        }
      />
    </div>
  );
}
