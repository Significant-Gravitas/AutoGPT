"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useState, type CSSProperties } from "react";
import { CARD } from "../../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { axisTicks, formatDuration, formatTick, type Trace } from "../trace";
import styles from "../trace.module.css";
import { TRACE_GRID, TraceRow } from "./TraceRow";

interface Props {
  trace: Trace;
}

/** The run as a waterfall: a row per block run in the order they started,
 *  each bar placed on one time axis, so slow and parallel blocks stand out.
 *  One row opens at a time. */
export function TraceWaterfall({ trace }: Props) {
  const [openID, setOpenID] = useState<string | null>(null);
  const ticks = axisTicks(trace.totalMs);

  return (
    <div className={cn(styles.trace, CARD, "flex flex-col p-1")}>
      <div aria-hidden className={cn(TRACE_GRID, "h-6 px-2")}>
        <span />
        <Text variant="small" as="span" tone="faint">
          Block
        </Text>
        <span className="relative h-full">
          {ticks.map((at) => (
            <Text
              key={at}
              variant="small"
              as="span"
              tone="faint"
              tabular
              className="absolute top-1/2 left-(--at) -translate-y-1/2"
              style={
                { "--at": `${(at / trace.totalMs) * 100}%` } as CSSProperties
              }
            >
              {formatTick(at)}
            </Text>
          ))}
        </span>
        <Text variant="small" as="span" tone="faint" align="right" tabular>
          {formatDuration(trace.totalMs)}
        </Text>
      </div>
      <ul aria-label="Block runs" className="flex flex-col">
        {trace.steps.map((step, index) => (
          <TraceRow
            key={step.id}
            step={step}
            index={index}
            totalMs={trace.totalMs}
            ticks={ticks}
            isOpen={openID === step.id}
            onToggle={() => setOpenID(openID === step.id ? null : step.id)}
          />
        ))}
      </ul>
    </div>
  );
}
