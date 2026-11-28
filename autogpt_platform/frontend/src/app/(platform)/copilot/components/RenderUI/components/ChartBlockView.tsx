import type { ChartBlock } from "@/app/api/__generated__/models/chartBlock";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import dynamic from "next/dynamic";
import { CHART_LEGEND_CLASSES, getChartSummary } from "../helpers";
import { BlockCard } from "./BlockCard";

const ChartCanvas = dynamic(
  () => import("./ChartCanvas").then((module) => module.ChartCanvas),
  { ssr: false, loading: () => <Skeleton className="h-full w-full" /> },
);

interface Props {
  block: ChartBlock;
}

export function ChartBlockView({ block }: Props) {
  return (
    <BlockCard title={block.title}>
      {block.series.length > 1 ? (
        <ul className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
          {block.series.map((series, index) => (
            <li key={index} className="flex items-center gap-1.5">
              <span
                aria-hidden
                className={`size-2.5 rounded-sm ${CHART_LEGEND_CLASSES[index]}`}
              />
              <Text variant="small" tone="secondary">
                {series.name}
              </Text>
            </li>
          ))}
        </ul>
      ) : null}
      <div
        role="img"
        aria-label={getChartSummary(block)}
        className="h-44 w-full"
      >
        <ChartCanvas block={block} />
      </div>
    </BlockCard>
  );
}
