import { useState } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { SeriesChart } from "@/lib/openui/catalog-dashboard";
import { SeriesPlot } from "./SeriesPlot";

export function SeriesChartView({
  props,
}: ComponentRenderProps<z.infer<typeof SeriesChart.props>>) {
  return <SeriesChartContent data={props} />;
}

interface Props {
  data: z.infer<typeof SeriesChart.props>;
}
export function SeriesChartContent({ data }: Props) {
  const [hidden, setHidden] = useState<string[]>([]);
  return (
    <section className="min-w-0 space-y-3" aria-label={data.title}>
      <h3 className="text-sm font-semibold">{data.title}</h3>
      <p className="text-xs text-muted-foreground">
        {data.description} {data.unit}
      </p>
      <div className="flex flex-wrap gap-3">
        {data.series.map((series) => (
          <label
            key={series.key}
            className="flex min-h-11 items-center gap-2 text-xs"
          >
            <input
              type="checkbox"
              checked={!hidden.includes(series.key)}
              onChange={(event) =>
                setHidden(
                  event.target.checked
                    ? hidden.filter((key) => key !== series.key)
                    : [...hidden, series.key],
                )
              }
            />
            {series.label}
          </label>
        ))}
      </div>
      <SeriesPlot
        data={data}
        visible={data.series
          .map((series) => series.key)
          .filter((key) => !hidden.includes(key))}
      />
      <details>
        <summary className="min-h-11 cursor-pointer py-3 text-xs">
          View chart data
        </summary>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr>
                <th className="p-2">Observation</th>
                {data.series.map((series) => (
                  <th key={series.key} className="p-2">
                    {series.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.points.map((point, index) => (
                <tr key={index}>
                  <th className="p-2 font-normal">{point.label}</th>
                  {data.series.map((series) => (
                    <td key={series.key} className="p-2">
                      {point.values[series.key] == null
                        ? "Unknown"
                        : point.values[series.key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}
