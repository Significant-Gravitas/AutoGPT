import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { TrendChart } from "@/lib/openui/catalog-sections";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { colors } from "@/components/styles/colors";
import { ChartHeading } from "./ChartHeading";

export function TrendChartView({
  props,
}: ComponentRenderProps<z.infer<typeof TrendChart.props>>) {
  const points = (props.points ?? [])
    .slice(0, 60)
    .filter((point) => point && Number.isFinite(point.value));
  return (
    <section
      className="min-w-0 rounded-xl border border-zinc-200 bg-white p-5"
      aria-label={props.title}
    >
      <ChartHeading {...props} />
      <div
        className="h-48 min-w-0"
        role="img"
        aria-label={`${props.title}: ${points.map((point) => `${point.label} ${point.value} ${props.unit}`).join(", ")}`}
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <LineChart
            data={points}
            margin={{ top: 8, right: 12, left: -18, bottom: 0 }}
            accessibilityLayer
          >
            <CartesianGrid vertical={false} stroke={colors.zinc[100]} />
            <XAxis
              dataKey="label"
              axisLine={false}
              tickLine={false}
              tick={{ fill: colors.zinc[500], fontSize: 11 }}
              dy={6}
              minTickGap={24}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: colors.zinc[400], fontSize: 11 }}
            />
            <Tooltip />
            <Line
              type="linear"
              dataKey="value"
              name={props.unit}
              stroke={colors.purple[500]}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
