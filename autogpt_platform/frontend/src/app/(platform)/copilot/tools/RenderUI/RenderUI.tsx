"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import type { ToolUIPart } from "ai";
import { ChartBlockView } from "./components/ChartBlockView";
import { GalleryBlockView } from "./components/GalleryBlockView";
import { OptionsBlockView } from "./components/OptionsBlockView";
import { StatsBlockView } from "./components/StatsBlockView";
import { TableBlockView } from "./components/TableBlockView";
import { TimelineBlockView } from "./components/TimelineBlockView";
import { WorkflowDiagramBlockView } from "./components/WorkflowDiagramBlockView";
import { parseUIBlocks, type UIBlock } from "./helpers";

interface Props {
  part: ToolUIPart;
  canAct: boolean;
  isCurrentlyStreaming?: boolean;
  fileUrlBuilder?: (fileId: string) => string;
}

export function RenderUI({
  part,
  canAct,
  isCurrentlyStreaming,
  fileUrlBuilder,
}: Props) {
  const blocks = parseUIBlocks(part);

  if (!blocks) {
    // A call still open when the stream is over never finished, so it must
    // not stay on screen as a loading card.
    const isPending =
      part.state === "input-streaming" || part.state === "input-available";
    if (!isCurrentlyStreaming || !isPending) return null;
    return (
      <Skeleton
        data-testid="render-ui-skeleton"
        className="my-2 h-32 w-full rounded-2xl"
      />
    );
  }

  return (
    <div className="my-2 flex w-full min-w-0 flex-col gap-2">
      {blocks.map((block, index) => (
        <BlockView
          key={index}
          block={block}
          canAct={canAct}
          fileUrlBuilder={fileUrlBuilder}
        />
      ))}
    </div>
  );
}

interface BlockViewProps {
  block: UIBlock;
  canAct: boolean;
  fileUrlBuilder?: (fileId: string) => string;
}

function BlockView({ block, canAct, fileUrlBuilder }: BlockViewProps) {
  switch (block.type) {
    case "stats":
      return <StatsBlockView block={block} />;
    case "chart":
      return <ChartBlockView block={block} />;
    case "workflow_diagram":
      return <WorkflowDiagramBlockView block={block} />;
    case "gallery":
      return <GalleryBlockView block={block} fileUrlBuilder={fileUrlBuilder} />;
    case "timeline":
      return <TimelineBlockView block={block} />;
    case "options":
      return <OptionsBlockView block={block} canAct={canAct} />;
    case "table":
      return <TableBlockView block={block} />;
  }
}
