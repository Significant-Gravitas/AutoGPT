import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { BotTimeseriesPoint } from "@/app/api/__generated__/models/botTimeseriesPoint";

import { formatDay } from "./helpers";

interface Props {
  data: BotTimeseriesPoint[];
}

export function MessageVolumeChart({ data }: Props) {
  const chartData = data.map((point) => ({
    date: formatDay(point.date),
    Messages: point.messages,
    Replies: point.replies,
    Errors: point.errors,
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart
        data={chartData}
        margin={{ top: 8, right: 24, left: 0, bottom: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" fontSize={12} />
        <YAxis allowDecimals={false} fontSize={12} />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="Messages"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="Replies"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="Errors"
          stroke="#ef4444"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
