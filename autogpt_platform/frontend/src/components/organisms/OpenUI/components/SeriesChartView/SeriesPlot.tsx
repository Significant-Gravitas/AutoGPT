import { colors as palette } from "@/components/styles/colors";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { z } from "zod/v4";
import type { SeriesChart } from "@/lib/openui/catalog-dashboard";

interface Props {
  data: z.infer<typeof SeriesChart.props>;
  visible: string[];
}
const colors = [
  palette.purple[500],
  palette.slate[500],
  palette.orange[500],
  palette.green[500],
  palette.red[500],
];

export function SeriesPlot({ data, visible }: Props) {
  const points = data.points.map((point) => ({
    label: point.label,
    ...point.values,
  }));
  const axes = (
    <>
      <CartesianGrid vertical={false} stroke="currentColor" opacity={0.1} />
      <XAxis dataKey="label" minTickGap={24} tick={{ fontSize: 11 }} />
      <YAxis tick={{ fontSize: 11 }} width={55} />
      <Tooltip />
    </>
  );
  return (
    <div
      className="h-64 min-w-0"
      role="img"
      aria-label={`${data.title}. ${data.series.map((series) => series.label).join(", ")} in ${data.unit}. Values are in the data table below.`}
    >
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        {data.kind === "line" ? (
          <LineChart data={points} accessibilityLayer>
            {axes}
            {data.series
              .filter((series) => visible.includes(series.key))
              .map((series) => (
                <Line
                  key={series.key}
                  type="linear"
                  dataKey={series.key}
                  name={series.label}
                  stroke={colors[data.series.indexOf(series)]}
                  strokeWidth={2}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              ))}
          </LineChart>
        ) : (
          <BarChart data={points} accessibilityLayer>
            {axes}
            {data.series
              .filter((series) => visible.includes(series.key))
              .map((series) => (
                <Bar
                  key={series.key}
                  dataKey={series.key}
                  name={series.label}
                  fill={colors[data.series.indexOf(series)]}
                  stackId={data.kind === "stacked" ? "values" : undefined}
                  isAnimationActive={false}
                />
              ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
