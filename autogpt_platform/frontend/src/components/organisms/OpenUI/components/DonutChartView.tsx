import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { DonutChart } from "@/lib/openui/catalog-sections";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { colors } from "@/components/styles/colors";
import { ChartHeading } from "./ChartHeading";

const palette = [
  colors.purple[500],
  colors.green[500],
  colors.orange[400],
  colors.zinc[500],
  colors.purple[300],
  colors.green[300],
];

export function DonutChartView({
  props,
}: ComponentRenderProps<z.infer<typeof DonutChart.props>>) {
  const points = (props.points ?? [])
    .slice(0, 12)
    .filter(
      (point) => point && Number.isFinite(point.value) && point.value >= 0,
    );
  const total = points.reduce((sum, point) => sum + point.value, 0);
  return (
    <section
      className="min-w-0 rounded-xl border border-zinc-200 bg-white p-5"
      aria-label={props.title}
    >
      <ChartHeading {...props} />
      <DonutPlot {...props} points={points} total={total} />
      <DonutLegend points={points} total={total} />
    </section>
  );
}

interface LegendProps {
  points: { label: string; value: number }[];
  total: number;
}

function DonutLegend({ points, total }: LegendProps) {
  return (
    <ul className="mt-3 divide-y divide-zinc-100 text-xs">
      {points.map((point, index) => (
        <li key={index} className="flex items-center gap-2 py-2">
          <svg className="size-2.5 shrink-0" viewBox="0 0 10 10" aria-hidden>
            <circle
              cx="5"
              cy="5"
              r="5"
              fill={palette[index % palette.length]}
            />
          </svg>
          <span className="min-w-0 flex-1 break-words text-zinc-600">
            {point.label}
          </span>
          <span className="shrink-0 font-medium text-zinc-800">
            {point.value.toLocaleString()} ·{" "}
            {total ? Math.round((point.value / total) * 100) : 0}%
          </span>
        </li>
      ))}
    </ul>
  );
}

interface PlotProps extends LegendProps {
  title: string;
  unit: string;
}

function DonutPlot({ points, total, title, unit }: PlotProps) {
  return total > 0 ? (
    <div
      className="relative h-48 min-w-0"
      role="img"
      aria-label={`${title}: ${points.map((point) => `${point.label} ${point.value} ${unit}`).join(", ")}`}
    >
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        <PieChart accessibilityLayer>
          <Pie
            data={points}
            dataKey="value"
            nameKey="label"
            innerRadius={58}
            outerRadius={84}
            paddingAngle={2}
            isAnimationActive={false}
          >
            {points.map((point, index) => (
              <Cell key={index} fill={palette[index % palette.length]} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xl font-semibold text-zinc-900">
          {total.toLocaleString()}
        </span>
        <span className="text-xs text-zinc-500">Total</span>
      </div>
    </div>
  ) : (
    <p className="py-8 text-center text-sm text-zinc-500">
      No positive amounts to chart.
    </p>
  );
}
