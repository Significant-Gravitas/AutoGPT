import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { BotServerCountPoint } from "@/app/api/__generated__/models/botServerCountPoint";

import { formatDay, formatNumber, SHARDING_THRESHOLD } from "./helpers";

interface Props {
  data: BotServerCountPoint[];
}

export function ServerGrowthChart({ data }: Props) {
  const chartData = data.map((point) => ({
    date: formatDay(point.date),
    Servers: point.server_count,
  }));
  const peak = data.reduce(
    (max, point) => Math.max(max, point.server_count),
    0,
  );
  // Only fold the (far-off) sharding threshold into the Y-axis once we're
  // actually approaching it — otherwise a handful of servers would render as a
  // flat line pinned to the bottom of a 0–2500 axis.
  const nearThreshold = peak > SHARDING_THRESHOLD * 0.4;

  return (
    <div>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart
          data={chartData}
          margin={{ top: 8, right: 24, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="serverFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" fontSize={12} />
          <YAxis
            allowDecimals={false}
            fontSize={12}
            domain={nearThreshold ? [0, SHARDING_THRESHOLD] : undefined}
          />
          <Tooltip />
          {nearThreshold && (
            <ReferenceLine
              y={SHARDING_THRESHOLD}
              stroke="#ef4444"
              strokeDasharray="4 4"
              label="Sharding threshold"
            />
          )}
          <Area
            type="monotone"
            dataKey="Servers"
            stroke="#6366f1"
            strokeWidth={2}
            fill="url(#serverFill)"
          />
        </AreaChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-muted-foreground">
        Sharding threshold: {formatNumber(SHARDING_THRESHOLD)} servers · current
        peak in range: {formatNumber(peak)}
      </p>
    </div>
  );
}
