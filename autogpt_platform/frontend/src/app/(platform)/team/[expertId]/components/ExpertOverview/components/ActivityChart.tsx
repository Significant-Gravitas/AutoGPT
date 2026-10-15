"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Text } from "@/components/atoms/Text/Text";
import { SERIES_COLORS, type ChartDay } from "../helpers";

interface Props {
  days: ChartDay[];
}

const SERIES = [
  { key: "sessions", label: "Conversations" },
  { key: "runs", label: "Workflow runs" },
] as const;

/** Conversations and workflow runs per day, stacked: both are counts on one
 *  scale, so one axis carries them. */
export function ActivityChart({ days }: Props) {
  return (
    <div className="flex flex-col gap-3">
      <ul className="flex items-center gap-4" aria-label="Legend">
        {SERIES.map((series) => (
          <li key={series.key} className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="size-2.5 rounded-sm"
              style={{ backgroundColor: SERIES_COLORS[series.key] }}
            />
            <Text variant="small" className="!text-zinc-600">
              {series.label}
            </Text>
          </li>
        ))}
      </ul>
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={days}
            margin={{ top: 4, right: 4, bottom: 0, left: -24 }}
          >
            <CartesianGrid vertical={false} stroke="#f0f0ef" />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={false}
              interval={6}
              tick={{ fontSize: 11, fill: "#71717a" }}
            />
            <YAxis
              allowDecimals={false}
              tickLine={false}
              axisLine={false}
              width={48}
              tick={{ fontSize: 11, fill: "#71717a" }}
            />
            <Tooltip cursor={{ fill: "#f4f4f5" }} content={<ChartTooltip />} />
            <Bar
              dataKey="sessions"
              name="Conversations"
              stackId="activity"
              fill={SERIES_COLORS.sessions}
              stroke="#ffffff"
              strokeWidth={2}
              maxBarSize={16}
            />
            <Bar
              dataKey="runs"
              name="Workflow runs"
              stackId="activity"
              fill={SERIES_COLORS.runs}
              stroke="#ffffff"
              strokeWidth={2}
              radius={[4, 4, 0, 0]}
              maxBarSize={16}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface TooltipProps {
  active?: boolean;
  payload?: { payload: ChartDay }[];
}

function ChartTooltip({ active, payload }: TooltipProps) {
  const day = active ? payload?.[0]?.payload : null;
  if (!day) return null;

  return (
    <div className="rounded-md bg-zinc-900 px-3 py-2 text-xs text-zinc-50 shadow-sm">
      <p className="mb-1 font-medium">{day.label}</p>
      {SERIES.map((series) => (
        <p key={series.key} className="flex items-center gap-1.5 tabular-nums">
          <span
            aria-hidden
            className="size-2 rounded-sm"
            style={{ backgroundColor: SERIES_COLORS[series.key] }}
          />
          {series.label}: {day[series.key]}
        </p>
      ))}
    </div>
  );
}
