"use client";

import React, { useState } from "react";
import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { Button } from "@/components/atoms/Button/Button";
import type { BlockOutputResponse } from "@/app/api/__generated__/models/blockOutputResponse";
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import type { OutputMetadata } from "@/components/contextual/OutputRenderers";
import {
  ContentBadge,
  ContentCard,
  ContentCardTitle,
  ContentGrid,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";

interface Props {
  output: BlockOutputResponse;
}

const COLLAPSED_LIMIT = 3;

function isWorkspaceRef(value: unknown): value is string {
  return typeof value === "string" && value.startsWith("workspace://");
}

function resolveForRenderer(value: unknown): {
  value: unknown;
  metadata?: OutputMetadata;
} {
  if (!isWorkspaceRef(value)) return { value };

  const withoutPrefix = value.replace("workspace://", "");
  const fileId = withoutPrefix.split("#")[0];
  const apiPath = getGetWorkspaceDownloadFileByIdUrl(fileId);
  const url = `/api/proxy${apiPath}`;

  const hashIndex = value.indexOf("#");
  const mimeHint =
    hashIndex !== -1 ? value.slice(hashIndex + 1) || undefined : undefined;

  const metadata: OutputMetadata = {};
  if (mimeHint) {
    metadata.mimeType = mimeHint;
    if (mimeHint.startsWith("image/")) metadata.type = "image";
    else if (mimeHint.startsWith("video/")) metadata.type = "video";
  }

  return { value: url, metadata };
}

function RenderOutputValue({ value }: { value: unknown }) {
  const resolved = resolveForRenderer(value);
  const renderer = globalRegistry.getRenderer(
    resolved.value,
    resolved.metadata,
  );

  if (renderer) {
    return (
      <OutputItem
        value={resolved.value}
        metadata={resolved.metadata}
        renderer={renderer}
      />
    );
  }

  // Fallback for audio workspace refs
  if (
    isWorkspaceRef(value) &&
    resolved.metadata?.mimeType?.startsWith("audio/")
  ) {
    return (
      <audio controls src={String(resolved.value)} className="mt-2 w-full" />
    );
  }

  return null;
}

function OutputKeySection({
  outputKey,
  items,
}: {
  outputKey: string;
  items: unknown[];
}) {
  const [expanded, setExpanded] = useState(false);
  const hasMoreItems = items.length > COLLAPSED_LIMIT;
  const visibleItems = expanded ? items : items.slice(0, COLLAPSED_LIMIT);

  return (
    <ContentCard>
      <div className="flex items-center justify-between gap-2">
        <ContentCardTitle className="text-xs">{outputKey}</ContentCardTitle>
        <ContentBadge>
          {items.length} item{items.length === 1 ? "" : "s"}
        </ContentBadge>
      </div>
      <div className="mt-2">
        {visibleItems.map((item, i) => (
          <RenderOutputValue key={i} value={item} />
        ))}
      </div>
      {hasMoreItems && (
        <Button
          variant="ghost"
          size="small"
          className="mt-1 h-auto px-0 py-0.5 text-[11px] text-muted-foreground"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? "Show less" : `Show all ${items.length} items`}
        </Button>
      )}
    </ContentCard>
  );
}

export function BlockOutputCard({ output }: Props) {
  return (
    <ContentGrid>
      <ContentMessage>{output.message}</ContentMessage>

      {Object.entries(output.outputs ?? {}).map(([key, items]) => (
        <OutputKeySection key={key} outputKey={key} items={items} />
      ))}
    </ContentGrid>
  );
}
