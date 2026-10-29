"use client";

import { Button } from "@/components/atoms/Button/Button";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { buildRunLink } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { useState } from "react";
import { type WorkRunMetadata } from "./helpers";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  metadata: WorkRunMetadata;
  preview: string;
}

export function WorkCard({ metadata, preview }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const runLink = buildRunLink(metadata.libraryAgentId, metadata.executionId);

  return (
    <div
      className="max-w-md rounded-2xl bg-surface p-4 ring-1 ring-inset ring-border-base/60"
      data-testid="work-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <Text variant="body-medium-tight" tone="primary" className="truncate">
            {metadata.graphName}
          </Text>
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
        <Text variant="body-tight" tone="muted" className="mt-2 line-clamp-2">
          {preview}
        </Text>
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
