"use client";

import type { ChartBlock } from "@/app/api/__generated__/models/chartBlock";
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getChartData } from "../helpers";

// Hex throughout: recharts writes these as SVG attributes, where CSS variables
// do not resolve (ds-colour-exempt.json). Series match CHART_LEGEND_CLASSES.
const SERIES_COLORS = ["#7733f5", "#0ea5e9", "#fe8700"];
const GRID_COLOR = "#efeff0";
const TICK_COLOR = "#83838c";
const ANNOTATION_COLOR = "#68686f";

interface Props {
  block: ChartBlock;
}

export function ChartCanvas({ block }: Props) {
  const annotation = block.annotation;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart
        data={getChartData(block)}
        margin={{ top: 16, right: 8, bottom: 0, left: -16 }}
      >
        <CartesianGrid vertical={false} stroke={GRID_COLOR} />
        <XAxis
          dataKey="label"
          tickLine={false}
          axisLine={false}
          minTickGap={16}
          tick={{ fontSize: 11, fill: TICK_COLOR }}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          unit={block.unit ?? undefined}
          tick={{ fontSize: 11, fill: TICK_COLOR }}
        />
        <Tooltip />
        {annotation ? (
          <ReferenceLine
            x={annotation.x}
            stroke={ANNOTATION_COLOR}
            strokeDasharray="3 3"
            label={{
              value: annotation.label,
              position: "top",
              fontSize: 11,
              fill: ANNOTATION_COLOR,
            }}
          />
        ) : null}
        {block.series.map((series, index) => {
          const shared = {
            dataKey: `series${index}`,
            name: series.name,
            stroke: SERIES_COLORS[index],
            isAnimationActive: false,
          };
          if (block.kind === "bar") {
            return (
              <Bar
                key={index}
                {...shared}
                fill={SERIES_COLORS[index]}
                maxBarSize={24}
                radius={[4, 4, 0, 0]}
              />
            );
          }
          if (block.kind === "area") {
            return (
              <Area
                key={index}
                {...shared}
                type="monotone"
                strokeWidth={2}
                fill={SERIES_COLORS[index]}
                fillOpacity={0.15}
              />
            );
          }
          return (
            <Line
              key={index}
              {...shared}
              type="monotone"
              strokeWidth={2}
              dot={false}
            />
          );
        })}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
