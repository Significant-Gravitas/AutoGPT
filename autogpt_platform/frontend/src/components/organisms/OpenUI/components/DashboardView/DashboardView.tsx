import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Dashboard } from "@/lib/openui/catalog-dashboard";
import { aggregateRows, plotPoints } from "@/lib/openui/dashboard";
import { downloadText, csvText } from "@/lib/openui/download";
import { Button } from "@/components/atoms/Button/Button";
import { SeriesChartContent } from "../SeriesChartView/SeriesChartView";
import { DashboardFilters } from "./DashboardFilters";
import { useDashboard } from "./useDashboard";

export function DashboardView({
  props,
}: ComponentRenderProps<z.infer<typeof Dashboard.props>>) {
  const dashboard = useDashboard(props);
  return (
    <section aria-label={props.title} className="min-w-0 space-y-4">
      <h3 className="text-base font-semibold">{props.title}</h3>
      <DashboardFilters
        columns={props.columns}
        dashboard={dashboard}
        bound={!!props.binding}
      />
      <p className="text-xs text-muted-foreground" aria-live="polite">
        {dashboard.filtered.length} of {dashboard.rows.length} rows · All
        sections use these filters
      </p>
      {dashboard.error ? (
        <p role="alert" className="text-sm text-destructive">
          {dashboard.error}
        </p>
      ) : (
        <DashboardContent props={props} dashboard={dashboard} />
      )}
    </section>
  );
}

interface Props {
  props: z.infer<typeof Dashboard.props>;
  dashboard: ReturnType<typeof useDashboard>;
}
function DashboardContent({ props, dashboard }: Props) {
  return (
    <>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {props.metrics.map((metric, index) => {
          const value = aggregateRows(dashboard.filtered, metric);
          return (
            <div key={index}>
              <p className="text-xs text-muted-foreground">{metric.label}</p>
              <output
                aria-live="polite"
                className="block text-xl font-semibold tabular-nums"
              >
                {value === null
                  ? "Unknown"
                  : new Intl.NumberFormat(undefined, {
                      maximumFractionDigits: 2,
                    }).format(value)}{" "}
                {metric.unit}
              </output>
            </div>
          );
        })}
      </div>
      {props.charts.map((chart, index) => (
        <SeriesChartContent
          key={index}
          data={{
            ...chart,
            description: "Filtered supplied observations",
            points: plotPoints(dashboard.filtered, chart),
          }}
        />
      ))}
      <details>
        <summary className="min-h-11 cursor-pointer py-3 text-sm">
          View filtered rows
        </summary>
        <div className="max-h-80 overflow-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr>
                {props.columns.map((column) => (
                  <th key={column.key} className="p-2">
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dashboard.filtered.map((row) => (
                <tr key={row.id}>
                  {props.columns.map((column) => (
                    <td key={column.key} className="p-2">
                      {row.cells[column.key] == null
                        ? "Unknown"
                        : String(row.cells[column.key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <Button
        type="button"
        variant="secondary"
        size="small"
        onClick={() =>
          downloadText(
            `${props.name}.csv`,
            csvText([
              props.columns.map((column) => column.label),
              ...dashboard.filtered.map((row) =>
                props.columns.map((column) => row.cells[column.key]),
              ),
            ]),
            "text/csv;charset=utf-8",
          )
        }
      >
        Download filtered data
      </Button>
    </>
  );
}
