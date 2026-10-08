import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Chart } from "@/lib/openui/catalog";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { colors } from "@/components/styles/colors";

export function ChartView({
  props,
}: ComponentRenderProps<z.infer<typeof Chart.props>>) {
  const points = (props.points ?? [])
    .slice(0, 24)
    .filter((point) => point && Number.isFinite(point.value));
  return (
    <section
      className="min-w-0 rounded-xl border border-zinc-200 bg-white p-5"
      aria-label={props.title}
    >
      <div className="mb-5 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-zinc-800">{props.title}</h3>
          <p className="mt-1 text-xs text-zinc-500">{props.description}</p>
        </div>
        <span className="rounded-md bg-purple-50 px-2 py-1 text-xs text-purple-700">
          {props.unit}
        </span>
      </div>
      <div
        className="h-44 min-w-0"
        role="img"
        aria-label={`${props.title}: ${points.map((point) => `${point.label} ${point.value} ${props.unit}`).join(", ")}`}
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <BarChart
            data={points}
            margin={{ top: 6, right: 0, left: -24, bottom: 0 }}
            accessibilityLayer
          >
            <CartesianGrid vertical={false} stroke={colors.zinc[100]} />
            <XAxis
              dataKey="label"
              axisLine={false}
              tickLine={false}
              tick={{ fill: colors.zinc[500], fontSize: 11 }}
              dy={6}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: colors.zinc[400], fontSize: 11 }}
            />
            <Tooltip cursor={{ fill: colors.purple[50] }} />
            <Bar
              dataKey="value"
              name={props.unit}
              fill={colors.purple[400]}
              radius={[5, 5, 0, 0]}
              maxBarSize={44}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
