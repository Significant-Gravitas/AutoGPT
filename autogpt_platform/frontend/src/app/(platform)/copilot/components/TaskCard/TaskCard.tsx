"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { buildRunLink } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { useState } from "react";
import { type TaskCardMetadata } from "./helpers";

interface Props {
  metadata: TaskCardMetadata;
  preview: string;
}

export function TaskCard({ metadata, preview }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const succeeded = metadata.status === "DONE";

  return (
    <div
      className="max-w-md rounded-2xl bg-white p-4 ring-1 ring-inset ring-zinc-200"
      data-testid="task-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-zinc-900">
            {metadata.graphName}
          </p>
          <div className="mt-1">
            <Badge variant={succeeded ? "success" : "error"} size="small">
              {succeeded ? "Task done" : "Task failed"}
            </Badge>
          </div>
        </div>
        <Button
          variant="secondary"
          size="small"
          onClick={() => setIsOpen(true)}
        >
          Open
        </Button>
      </div>
      {preview ? (
        <p className="mt-2 line-clamp-2 text-sm text-zinc-500">{preview}</p>
      ) : null}

      <WorkOutputSheet
        open={isOpen}
        onOpenChange={setIsOpen}
        title={metadata.graphName}
        // The outcome post carries no output classification, so the sheet
        // opens on its run-link fallback rather than guessing a viewer.
        outputType="unknown"
        outputKey={null}
        graphId={metadata.graphId}
        executionId={metadata.executionId}
        runLink={buildRunLink(metadata.libraryAgentId, metadata.executionId)}
      />
    </div>
  );
}
