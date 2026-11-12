import type { ChartBlock } from "@/app/api/__generated__/models/chartBlock";
import type { TableBlock } from "@/app/api/__generated__/models/tableBlock";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { UIBlocksResponse } from "@/app/api/__generated__/models/uIBlocksResponse";
import type { Tone } from "@/components/styles/tones";
import type { ToolUIPart } from "ai";

export const RENDER_UI_PART_TYPE = "tool-render_ui";

export type UIBlock = NonNullable<UIBlocksResponse["blocks"]>[number];
export type BlockTone = "neutral" | "positive" | "negative" | "warning";

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isObjectList(value: unknown) {
  return Array.isArray(value) && value.every(isObject);
}

// What each block's view reads as a list, so a malformed stored block is
// skipped instead of breaking the cards next to it. A type missing here is
// one this tab is too old to know, and it is skipped too.
const BLOCK_CHECKS: Record<
  string,
  (block: Record<string, unknown>) => boolean
> = {
  stats: (block) => isObjectList(block.items),
  chart: (block) =>
    Array.isArray(block.x_labels) &&
    isObjectList(block.series) &&
    (block.series as Record<string, unknown>[]).every((series) =>
      Array.isArray(series.values),
    ),
  workflow_diagram: (block) => isObjectList(block.steps),
  gallery: (block) => isObjectList(block.images),
  timeline: (block) => isObjectList(block.items),
  options: (block) => isObjectList(block.items),
  table: (block) =>
    Array.isArray(block.columns) &&
    Array.isArray(block.rows) &&
    block.rows.every(Array.isArray),
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
  return isObject(value) ? value : null;
}

function isRenderable(value: unknown): value is UIBlock {
  if (!isObject(value)) return false;
  const check = BLOCK_CHECKS[String(value.type)];
  return Boolean(check) && check(value);
}

export function parseUIBlocks(part: ToolUIPart): UIBlock[] | null {
  const output = toRecord(part.output);
  if (!output || output.type !== ResponseType.ui_blocks) return null;
  if (!Array.isArray(output.blocks)) return null;
  const blocks = output.blocks.filter(isRenderable);
  return blocks.length > 0 ? blocks : null;
}

/** What a screen reader hears for a chart: each point with its label. */
export function getChartSummary(block: ChartBlock) {
  const unit = block.unit ? ` ${block.unit}` : "";
  const lines = block.series.map((series) => {
    const points = block.x_labels.map(
      (label, index) => `${label}: ${series.values[index]}${unit}`,
    );
    return `${series.name} — ${points.join(", ")}`;
  });
  const note = block.annotation
    ? [`Note at ${block.annotation.x}: ${block.annotation.label}`]
    : [];
  return [block.title ?? "Chart", ...lines, ...note].join(". ");
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

// A number, money or a percentage, optionally signed: "1,240", "$3.5k", "-12%".
const NUMERIC_CELL = /^[-+]?[$€£]?\d[\d,.]*\s?(%|[kKmMbB])?$/;

/** Columns whose every cell is a number, so they line up on the right. */
export function getNumericColumns(block: TableBlock) {
  return block.columns.map((_column, index) =>
    block.rows.every((row) => NUMERIC_CELL.test((row[index] ?? "").trim())),
  );
}
