"use client";

import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { InfoMeta } from "../../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "../../blocks";
import { formatWhen, runElapsedMs, statusView, triggerLabel } from "../runs";
import { formatDuration, type StepState } from "../trace";
import { StatusMark } from "./StatusMark";

interface Props {
  runs: GraphExecutionMeta[];
  now: number;
  onSelect: (executionID: string) => void;
}

const TONE_STATE: Record<string, StepState> = {
  live: "running",
  ok: "done",
  failed: "failed",
  idle: "stopped",
  waiting: "waiting",
};

/** The workflow's runs, newest first: how each started, when, how long it
 *  took and how it ended. A row opens that run on its own page of the tab. */
export function RunsList({ runs, now, onSelect }: Props) {
  return (
    <>
      {runs.map((run) => {
        const view = statusView(run.status);
        const elapsed = runElapsedMs(run, now);
        return (
          <BlockRow
            key={run.id}
            tile={<StatusMark state={TONE_STATE[view.tone]} />}
            title={triggerLabel(run)}
            subtitle={[
              formatWhen(run.started_at),
              elapsed !== null ? formatDuration(elapsed) : null,
            ]
              .filter(Boolean)
              .join(" · ")}
            trailing={<InfoMeta>{view.label}</InfoMeta>}
            onClick={() => onSelect(run.id)}
          />
        );
      })}
    </>
  );
}
