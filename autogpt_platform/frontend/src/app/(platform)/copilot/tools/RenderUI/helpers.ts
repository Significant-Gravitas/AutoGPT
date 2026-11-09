import type { ChartBlock } from "@/app/api/__generated__/models/chartBlock";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { UIBlocksResponse } from "@/app/api/__generated__/models/uIBlocksResponse";
import type { Tone } from "@/components/styles/tones";
import type { ToolUIPart } from "ai";

export const RENDER_UI_PART_TYPE = "tool-render_ui";

export type UIBlock = NonNullable<UIBlocksResponse["blocks"]>[number];
export type BlockTone = "neutral" | "positive" | "negative" | "warning";

// A block's type and the list it can't render without. A type missing here is
// one this tab is too old to know, and it is skipped.
const REQUIRED_LIST: Record<string, string> = {
  stats: "items",
  chart: "series",
  workflow_diagram: "steps",
  gallery: "images",
  timeline: "items",
  options: "items",
  table: "rows",
};

export const TEXT_TONE: Record<BlockTone, Tone> = {
  neutral: "muted",
  positive: "success",
  negative: "error-strong",
  warning: "warning",
};

export const CHART_LEGEND_CLASSES = ["bg-brand", "bg-sky", "bg-orange"];

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      return toRecord(JSON.parse(value) as unknown);
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function isRenderable(value: unknown): value is UIBlock {
  const block = toRecord(value);
  if (typeof value !== "object" || !block) return false;
  const list = REQUIRED_LIST[String(block.type)];
  return Boolean(list) && Array.isArray(block[list]);
}

export function parseUIBlocks(part: ToolUIPart): UIBlock[] | null {
  const output = toRecord(part.output);
  if (!output || output.type !== ResponseType.ui_blocks) return null;
  if (!Array.isArray(output.blocks)) return null;
  const blocks = output.blocks.filter(isRenderable);
  return blocks.length > 0 ? blocks : null;
}

export function getChartSummary(block: ChartBlock) {
  const lines = block.series.map(
    (series) => `${series.name}: ${series.values.join(", ")}`,
  );
  return [block.title ?? "Chart", ...lines].join(". ");
}

export function getChartData(block: ChartBlock) {
  return block.x_labels.map((label, pointIndex) => ({
    label,
    ...Object.fromEntries(
      block.series.map((series, seriesIndex) => [
        `series${seriesIndex}`,
        series.values[pointIndex],
      ]),
    ),
  }));
}
