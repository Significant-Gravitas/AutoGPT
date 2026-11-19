"use client";

import { useMemo, useState } from "react";

import {
  useGetV1ListExecutionSchedulesForAUser,
  useListUpcomingScheduleRuns,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";

import { buildSchedules } from "./helpers";
import { useScheduleTabs } from "./useScheduleTabs";

export const TASK_LIST_PARAMS = { limit: 100, include_subtasks: false };

/** Every schedule the user has: repeating, watching and once-only tasks, and
 *  workflow schedules, with their coming run times. */
export function useSchedulesPage() {
  const [loadedAt] = useState(() => Date.now());
  const tabs = useScheduleTabs();

  const tasksQuery = useListTasks(TASK_LIST_PARAMS, {
    query: { select: okData },
  });
  const workflowsQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: okData },
  });
  const upcomingQuery = useListUpcomingScheduleRuns(undefined, {
    query: { select: okData },
  });

  const tasks = tasksQuery.data?.tasks;
  const workflows = workflowsQuery.data;
  const upcoming = upcomingQuery.data?.schedules;
  // The Affect UI block keeps its own copy of the list and resets it whenever
  // the array changes, so it has to be the same array until the data does.
  const schedules = useMemo(
    () =>
      buildSchedules({
        tasks: tasks ?? [],
        workflows: workflows ?? [],
        upcoming: upcoming ?? [],
        now: loadedAt,
      }),
    [tasks, workflows, upcoming, loadedAt],
  );

  // The coming times only add to the list: it shows without them, each
  // schedule with its next run, and fills in when they arrive.
  const isLoading = tasksQuery.isLoading || workflowsQuery.isLoading;
  const error = tasksQuery.error ?? workflowsQuery.error ?? null;

  return {
    schedules,
    tasks: tasks ?? [],
    workflows: workflows ?? [],
    tabs,
    isLoading,
    error,
    isEmpty: !isLoading && !error && schedules.length === 0,
    retry: () => {
      tasksQuery.refetch();
      workflowsQuery.refetch();
      upcomingQuery.refetch();
    },
  };
}
