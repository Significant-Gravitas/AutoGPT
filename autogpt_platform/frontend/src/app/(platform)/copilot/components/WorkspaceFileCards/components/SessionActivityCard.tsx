"use client";

import { Clock01Icon, PlayIcon, Task01Icon } from "@hugeicons/core-free-icons";
import type { SessionRun, SessionSchedule, SessionTask } from "../helpers";
import { RunsList, SchedulesList, TasksList } from "./SessionActivityContent";
import { StackSection } from "./StackSection";

interface Props {
  runs: SessionRun[];
  schedules: SessionSchedule[];
  tasks: SessionTask[];
}

/**
 * What this chat set in motion, stacked under the workspace files: one card
 * per concern — tasks it delegated, runs it triggered, schedules it created —
 * so a single card doesn't read as one list.
 */
export function SessionActivityCard({ runs, schedules, tasks }: Props) {
  return (
    <>
      {tasks.length > 0 && (
        <StackSection title="Tasks" icon={Task01Icon} count={tasks.length}>
          <TasksList tasks={tasks} />
        </StackSection>
      )}
      {runs.length > 0 && (
        <StackSection title="Runs" icon={PlayIcon} count={runs.length}>
          <RunsList runs={runs} />
        </StackSection>
      )}
      {schedules.length > 0 && (
        <StackSection
          title="Schedules"
          icon={Clock01Icon}
          count={schedules.length}
        >
          <SchedulesList schedules={schedules} />
        </StackSection>
      )}
    </>
  );
}
