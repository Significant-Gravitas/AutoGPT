"use client";

import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import isEqual from "lodash/isEqual";
import { memo } from "react";
import { TaskListItem } from "./TaskListItem";

interface Props {
  run: GraphExecutionMeta;
  agent: LibraryAgent;
  selected: boolean;
  onSelectRun: (id: string, tab: "runs") => void;
  onRunDeleted: (id: string) => void;
}

function TaskListRowComponent({
  run,
  agent,
  selected,
  onSelectRun,
  onRunDeleted,
}: Props) {
  return (
    <div className="w-[15rem] lg:w-full">
      <TaskListItem
        run={run}
        title={agent.name}
        agent={agent}
        selected={selected}
        onClick={() => onSelectRun(run.id, "runs")}
        onDeleted={() => onRunDeleted(run.id)}
      />
    </div>
  );
}

// A refetch after a run is prepended hands every row a new `run` object even
// when its fields are unchanged, so the run is compared by value.
function areRowPropsEqual(prev: Props, next: Props) {
  return (
    prev.agent === next.agent &&
    prev.selected === next.selected &&
    prev.onSelectRun === next.onSelectRun &&
    prev.onRunDeleted === next.onRunDeleted &&
    isEqual(prev.run, next.run)
  );
}

export const TaskListRow = memo(TaskListRowComponent, areRowPropsEqual);
