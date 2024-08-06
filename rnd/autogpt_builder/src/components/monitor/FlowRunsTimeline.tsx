import { GraphMeta } from "@/lib/autogpt-server-api";
import {
  ComposedChart,
  DefaultLegendContentProps,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import moment from "moment/moment";
import { Card } from "@/components/ui/card";
import { cn, hashString } from "@/lib/utils";
import React from "react";
import { FlowRun } from "@/lib/types";
import { FlowRunStatusBadge } from "@/components/monitor/FlowRunStatusBadge";

export const FlowRunsTimeline = ({
  flows,
  flowRuns,
  dataMin,
  className,
}: {
  flows: GraphMeta[];
  flowRuns: FlowRun[];
  dataMin: "dataMin" | number;
  className?: string;
}) => (
  /* TODO: make logarithmic? */
  <ResponsiveContainer width="100%" height={120} className={className}>
    <ComposedChart>
      <XAxis
        dataKey="time"
        type="number"
        domain={[
          typeof dataMin == "string"
            ? dataMin
            : dataMin < 0
              ? Date.now() + dataMin * 1000
              : dataMin,
          Date.now(),
        ]}
        allowDataOverflow={true}
        tickFormatter={(unixTime) => {
          const now = moment();
          const time = moment(unixTime);
          return now.diff(time, "hours") < 24
            ? time.format("HH:mm")
            : time.format("YYYY-MM-DD HH:mm");
        }}
        name="Time"
        scale="time"
      />
      <YAxis
        dataKey="_duration"
        name="Duration (s)"
        tickFormatter={(s) => (s > 90 ? `${Math.round(s / 60)}m` : `${s}s`)}
      />
      <Tooltip
        content={({ payload, label }) => {
          if (payload && payload.length) {
            const data: FlowRun & { time: number; _duration: number } =
              payload[0].payload;
            const flow = flows.find((f) => f.id === data.graphID);
            return (
              <Card className="p-2 text-xs leading-normal">
                <p>
                  <strong>Agent:</strong> {flow ? flow.name : "Unknown"}
                </p>
                <p>
                  <strong>Status:</strong>&nbsp;
                  <FlowRunStatusBadge
                    status={data.status}
                    className="px-1.5 py-0"
                  />
                </p>
                <p>
                  <strong>Started:</strong>{" "}
                  {moment(data.startTime).format("YYYY-MM-DD HH:mm:ss")}
                </p>
                <p>
                  <strong>Duration / run time:</strong>{" "}
                  {formatDuration(data.duration)} /{" "}
                  {formatDuration(data.totalRunTime)}
                </p>
              </Card>
            );
          }
          return null;
        }}
      />
      {flows.map((flow) => (
        <Scatter
          key={flow.id}
          data={flowRuns
            .filter((fr) => fr.graphID == flow.id)
            .map((fr) => ({
              ...fr,
              time: fr.startTime + fr.totalRunTime * 1000,
              _duration: fr.totalRunTime,
            }))}
          name={flow.name}
          fill={`hsl(${(hashString(flow.id) * 137.5) % 360}, 70%, 50%)`}
        />
      ))}
      {flowRuns.map((run) => (
        <Line
          key={run.id}
          type="linear"
          dataKey="_duration"
          data={[
            { ...run, time: run.startTime, _duration: 0 },
            { ...run, time: run.endTime, _duration: run.totalRunTime },
          ]}
          stroke={`hsl(${(hashString(run.graphID) * 137.5) % 360}, 70%, 50%)`}
          strokeWidth={2}
          dot={false}
          legendType="none"
        />
      ))}
      <Legend
        content={<ScrollableLegend />}
        wrapperStyle={{
          bottom: 0,
          left: 0,
          right: 0,
          width: "100%",
          display: "flex",
          justifyContent: "center",
        }}
      />
    </ComposedChart>
  </ResponsiveContainer>
);

const ScrollableLegend: React.FC<
  DefaultLegendContentProps & { className?: string }
> = ({ payload, className }) => {
  return (
    <div
      className={cn(
        "whitespace-nowrap px-4 text-sm overflow-x-auto space-x-3",
        className,
      )}
      style={{ scrollbarWidth: "none" }}
    >
      {payload?.map((entry, index) => {
        if (entry.type == "none") return;
        return (
          <span key={`item-${index}`} className="inline-flex items-center">
            <span
              className="size-2.5 inline-block mr-1 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span>{entry.value}</span>
          </span>
        );
      })}
    </div>
  );
};

function formatDuration(seconds: number): string {
  return (
    (seconds < 100 ? seconds.toPrecision(2) : Math.round(seconds)).toString() +
    "s"
  );
}
