import type { z } from "zod/v4";
import type { dashboardMetric, dashboardPlot } from "./catalog-dashboard";
import type { DataRow } from "./data-values";
import { asNumber } from "./calculations";

export function aggregateRows(
  rows: DataRow[],
  metric: z.infer<typeof dashboardMetric>,
) {
  if (metric.aggregate === "count") return rows.length;
  if (!rows.length) return null;
  const values = rows.map((row) => asNumber(row.cells[metric.column]));
  if (values.some((value) => value === null)) return null;
  const numbers = values.filter((value): value is number => value !== null);
  if (metric.aggregate === "min") return Math.min(...numbers);
  if (metric.aggregate === "max") return Math.max(...numbers);
  const sum = numbers.reduce((a, b) => a + b, 0);
  const result = metric.aggregate === "average" ? sum / numbers.length : sum;
  return Number.isFinite(result) ? result : null;
}

export function plotPoints(
  rows: DataRow[],
  plot: z.infer<typeof dashboardPlot>,
) {
  return rows.map((row) => ({
    label: String(row.cells[plot.x] ?? "Unknown"),
    values: Object.fromEntries(
      plot.series.map((series) => [
        series.key,
        asNumber(row.cells[series.key]),
      ]),
    ),
  }));
}
