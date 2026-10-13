import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import {
  columnsSchema,
  identifier,
  rowsSchema,
  shortText,
  validateRows,
  calculatedRows,
  type DataColumn,
  type DataRow,
} from "./data-values";

export const chartSeries = z.object({ key: identifier, label: shortText });
export const plotKind = z.enum(["line", "bar", "stacked"]);
export const SeriesChart = defineComponent({
  name: "SeriesChart",
  description:
    "Multi-series chart with legend toggles, accessible data table, and true null gaps. kind is line/bar/stacked. All series use the same stated unit; stacked requires nonnegative values. Points preserve supplied order; each contains label and values keyed by series key, with null for a missing observation. Never connect across a missing period by dropping its point. Up to five series and 100 points.",
  props: z
    .object({
      title: shortText,
      description: z.string().max(1000),
      unit: z.string().max(50),
      kind: plotKind,
      series: z.array(chartSeries).min(1).max(5),
      points: z
        .array(
          z.object({
            label: shortText,
            values: z.record(identifier, z.number().nullable()),
          }),
        )
        .max(100),
    })
    .superRefine(({ series, points, kind }, context) => {
      const keys = series.map((item) => item.key);
      if (new Set(keys).size !== keys.length)
        context.addIssue({
          code: "custom",
          message: "Series keys must be unique.",
        });
      for (const point of points) {
        if (
          Object.keys(point.values).some((key) => !keys.includes(key)) ||
          keys.some((key) => !(key in point.values))
        )
          context.addIssue({
            code: "custom",
            message:
              "Each point must include every series key, using null for unknown observations.",
          });
        if (
          kind === "stacked" &&
          Object.values(point.values).some(
            (value) => value != null && value < 0,
          )
        )
          context.addIssue({
            code: "custom",
            message: "Use grouped bars or lines for signed values.",
          });
      }
    }),
  component: null,
});

export const dashboardPlot = z.object({
  title: shortText,
  kind: plotKind,
  x: identifier,
  series: z.array(chartSeries).min(1).max(5),
  unit: z.string().max(50),
});
export const dashboardMetric = z.object({
  label: shortText,
  column: identifier,
  aggregate: z.enum(["sum", "average", "min", "max", "count"]),
  unit: z.string().max(50),
});

export function chartDataError(
  charts: z.infer<typeof dashboardPlot>[],
  columns: DataColumn[],
  rows: DataRow[],
) {
  for (const chart of charts) {
    if (
      new Set(chart.series.map((series) => series.key)).size !==
      chart.series.length
    )
      return "Chart series must have distinct column keys.";
    if (
      chart.series.some((series) => {
        const column = columns.find((column) => column.key === series.key);
        return column?.unit && column.unit !== chart.unit;
      })
    )
      return "Chart series must use the same stated unit. Use separate charts for different units.";
    if (
      chart.kind === "stacked" &&
      rows.some((row) =>
        chart.series.some(
          (series) =>
            typeof row.cells[series.key] === "number" &&
            Number(row.cells[series.key]) < 0,
        ),
      )
    )
      return "Stacked charts require nonnegative values. Ask for grouped bars or lines for signed values.";
  }
  return "";
}
export const Dashboard = defineComponent({
  name: "Dashboard",
  description:
    "A linked analytical view with one shared typed dataset, text/category/numeric filters, calculated metrics, multi-series charts, table and CSV download. Every section uses the same filtered rows. Optional binding {form,field} reads the live rows of a DataGrid in this Workspace; supply matching columns and no duplicate rows when bound. Otherwise supply tool/user data (up to100 rows). Numeric unknowns pause the affected aggregate; missing observations remain gaps. Metrics support sum/average/min/max/count. Charts refer to actual column keys. Keep headings neutral; no stale totals in prose. This is a data snapshot, not a live account connection.",
  props: z
    .object({
      name: identifier,
      title: shortText,
      columns: columnsSchema,
      rows: rowsSchema,
      charts: z.array(dashboardPlot).max(4),
      metrics: z.array(dashboardMetric).max(6),
      binding: z.object({ form: identifier, field: identifier }).optional(),
    })
    .superRefine((props, context) => {
      for (const message of validateRows(props.columns, props.rows))
        context.addIssue({ code: "custom", message });
      const chartError = chartDataError(
        props.charts,
        props.columns,
        calculatedRows(props.columns, props.rows),
      );
      if (chartError) context.addIssue({ code: "custom", message: chartError });
      for (const metric of props.metrics) {
        const column = props.columns.find(
          (column) => column.key === metric.column,
        );
        if (
          metric.aggregate !== "count" &&
          column?.unit &&
          column.unit !== metric.unit
        )
          context.addIssue({
            code: "custom",
            message: "Metric units must match the source column.",
          });
      }
      for (const metric of props.metrics)
        if (
          !props.columns.some(
            (column) =>
              column.key === metric.column &&
              (metric.aggregate === "count" || column.type === "number"),
          )
        )
          context.addIssue({
            code: "custom",
            message: "Metrics must reference a column of the appropriate type.",
          });
      for (const chart of props.charts) {
        if (
          !props.columns.some((column) => column.key === chart.x) ||
          chart.series.some(
            (series) =>
              !props.columns.some(
                (column) =>
                  column.key === series.key && column.type === "number",
              ),
          )
        )
          context.addIssue({
            code: "custom",
            message:
              "Charts need an existing x column and numeric series columns.",
          });
      }
      if (props.binding && props.rows.length)
        context.addIssue({
          code: "custom",
          message:
            "A bound dashboard uses its grid's rows. Pass [] here to avoid stale duplicates.",
        });
    }),
  component: null,
});
