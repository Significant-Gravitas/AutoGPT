"use client";

import React, { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import type { BlockOutputResponse } from "@/app/api/__generated__/models/blockOutputResponse";
import {
  globalRegistry,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import { resolveForRenderer } from "@/app/(platform)/copilot/tools/ViewAgentOutput/ViewAgentOutput";
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
      {output.is_dry_run && (
        <div className="flex items-center gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs font-medium text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400">
          <span>⚡</span>
          <span>Simulated — no real execution occurred</span>
        </div>
      )}
      <ContentMessage>{output.message}</ContentMessage>

      {Object.entries(output.outputs ?? {}).map(([key, items]) => (
        <OutputKeySection key={key} outputKey={key} items={items} />
      ))}
    </ContentGrid>
  );
}
