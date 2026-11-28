import { ChartBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/ChartBlockView";
import { GalleryBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/GalleryBlockView";
import { StatsBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/StatsBlockView";
import {
  TableBlockView,
  type TableRowAction,
} from "@/app/(platform)/copilot/tools/RenderUI/components/TableBlockView";
import { TimelineBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/TimelineBlockView";
import { WorkflowDiagramBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/WorkflowDiagramBlockView";
import type { PageCards } from "@/app/api/__generated__/models/pageCards";
import type { PageLatestResult } from "@/app/api/__generated__/models/pageLatestResult";
import { Text } from "@/components/atoms/Text/Text";
import { formatDate } from "@/lib/utils/time";

type Block =
  | PageCards["blocks"][number]
  | NonNullable<PageLatestResult["blocks"]>[number];

interface Props {
  section: PageCards;
}

export function CardsSection({ section }: Props) {
  return (
    <div className="flex min-w-0 flex-col gap-3">
      {section.blocks.map((block, index) => (
        <BlockView key={index} block={block} />
      ))}
      {section.written_on ? (
        <Text variant="small" tone="secondary">
          Written by Otto on {formatDate(new Date(section.written_on))}
        </Text>
      ) : null}
    </div>
  );
}

export function BlockView({
  block,
  rowAction,
}: {
  block: Block;
  rowAction?: TableRowAction;
}) {
  switch (block.type) {
    case "stats":
      return <StatsBlockView block={block} />;
    case "table":
      return <TableBlockView block={block} rowAction={rowAction} />;
    case "chart":
      return <ChartBlockView block={block} />;
    case "gallery":
      return <GalleryBlockView block={block} />;
    case "timeline":
      return <TimelineBlockView block={block} />;
    case "workflow_diagram":
      return <WorkflowDiagramBlockView block={block} />;
  }
}
