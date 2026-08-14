"use client";

import { Button } from "@/components/atoms/Button/Button";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { buildRunLink } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { useState } from "react";
import { type WorkRunMetadata } from "./helpers";

interface Props {
  metadata: WorkRunMetadata;
  preview: string;
}

export function WorkCard({ metadata, preview }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const runLink = buildRunLink(metadata.libraryAgentId, metadata.executionId);

  return (
    <div
      className="max-w-md rounded-2xl bg-white p-4 ring-1 ring-inset ring-zinc-200"
      data-testid="work-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-zinc-900">
            {metadata.graphName}
          </p>
          <div className="mt-1">
            <RunStatusBadge status={metadata.status} />
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
        outputType={metadata.outputType}
        outputKey={metadata.outputKey}
        graphId={metadata.graphId}
        executionId={metadata.executionId}
        runLink={runLink}
      />
    </div>
  );
}
